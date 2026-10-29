import argparse
import os
import shutil
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .attack import calibrate_input_latent_bridge, collect_feature_stats
from .datasets import get_dataloaders
from .checkpoints import load_clean_checkpoint
from .models import build_split_model
from .training import (
    assign_cached_poison_masks,
    cache_smashed_features,
    evaluate_bridge,
    evaluate_clean,
    evaluate_input_attack,
    evaluate_latent_attack,
    evaluate_latent_masked_overwrite_attack,
    evaluate_latent_overwrite_attack,
    evaluate_target_class_latent_asr,
    split_train_epoch_backdoor,
    split_train_epoch_clean,
    train_server_epoch_from_cached_features,
)
from .utils import ensure_dir, get_device, load_config, save_json, set_seed


def merge_override(cfg: dict, args: argparse.Namespace) -> dict:
    if args.data_dir is not None:
        cfg['dataset']['data_dir'] = os.path.abspath(os.path.expanduser(args.data_dir))
    if args.clean_checkpoint is not None:
        cfg.setdefault('warmup_checkpoint', {})['load_path'] = os.path.abspath(os.path.expanduser(args.clean_checkpoint))
    if args.device is not None:
        cfg['device'] = args.device
    if args.num_workers is not None:
        cfg['dataset']['num_workers'] = args.num_workers
    if args.batch_size is not None:
        cfg['dataset']['batch_size'] = args.batch_size
    if args.no_download:
        cfg['dataset']['download'] = False
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.target_class is not None:
        cfg['attack']['target_class'] = args.target_class
    if args.split is not None:
        cfg['model']['split'] = args.split
    if args.arch is not None:
        cfg['model']['arch'] = args.arch
    if args.trigger_name is not None:
        cfg['attack']['trigger_name'] = args.trigger_name
    if args.warmup_epochs is not None:
        cfg['train']['warmup_epochs'] = args.warmup_epochs
    if args.attack_epochs is not None:
        cfg['train']['attack_epochs'] = args.attack_epochs
    return cfg


def make_optimizer(params, cfg: dict, lr: float):
    opt_name = cfg['train'].get('optimizer', 'adamw').lower()
    wd = float(cfg['train'].get('weight_decay', 0.0))
    if opt_name == 'sgd':
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(cfg['train'].get('momentum', 0.9)),
            weight_decay=wd,
            nesterov=bool(cfg['train'].get('nesterov', True)),
        )
    if opt_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f'Unsupported optimizer: {opt_name}')


def freeze_client_for_blackbox_queries(client: torch.nn.Module) -> None:
    client.eval()
    for p in client.parameters():
        p.requires_grad_(False)


def eval_all(client, server, loader, calib, cfg, device) -> Dict[str, float]:
    target = int(cfg['attack']['target_class'])
    out = {}
    out.update({f'clean_{k}': v for k, v in evaluate_clean(client, server, loader, device).items()})
    out.update(evaluate_input_attack(client, server, loader, calib.trigger, target, device))
    out.update(evaluate_latent_attack(
        client, server, loader, calib.delta, calib.cov_diag, target, cfg, device, shift_bank=calib.shift_bank
    ))
    out.update(evaluate_latent_overwrite_attack(client, server, loader, calib.delta, target, device))
    out.update(evaluate_latent_masked_overwrite_attack(client, server, loader, calib.delta, target, device))
    out.update(evaluate_bridge(client, loader, calib.trigger, calib.delta, device))
    out['delta_norm'] = float(calib.delta.norm().item())
    out['raw_delta_norm'] = float(getattr(calib, 'raw_delta_norm', 0.0))
    out['filtered_delta_norm'] = float(getattr(calib, 'filtered_delta_norm', 0.0))
    return out


def _coalesce_columns(df: pd.DataFrame, candidates, default=None) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype='object')
    for col in candidates:
        if col in df.columns:
            out = out.where(out.notna(), df[col])
    return out


def build_metrics_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(index=df.index)
    summary['stage'] = df['stage'] if 'stage' in df.columns else pd.NA
    summary['epoch'] = df['epoch'] if 'epoch' in df.columns else pd.NA
    summary['clean_acc'] = _coalesce_columns(df, ['clean_acc', 'test_acc'])
    summary['clean_loss'] = _coalesce_columns(df, ['clean_loss', 'test_loss'])
    summary['train_acc'] = _coalesce_columns(df, ['train_acc'])
    summary['train_loss'] = _coalesce_columns(df, ['train_loss'])
    summary['input_asr_all'] = _coalesce_columns(df, ['input_asr_all'])
    summary['input_asr_non_target'] = _coalesce_columns(df, ['input_asr_non_target'])
    summary['latent_asr'] = _coalesce_columns(df, ['latent_asr'])
    summary['latent_masked_overwrite_asr'] = _coalesce_columns(df, ['latent_masked_overwrite_asr'])
    summary['latent_overwrite_asr'] = _coalesce_columns(df, ['latent_overwrite_asr'])
    summary['bridge_cos'] = _coalesce_columns(df, ['bridge_cos'])
    summary['bridge_rel_error'] = _coalesce_columns(df, ['bridge_rel_error'])
    summary['trigger_shift_norm'] = _coalesce_columns(df, ['trigger_shift_norm'])
    summary['delta_norm'] = _coalesce_columns(df, ['delta_norm', 'delta_norm_eval'])
    summary['raw_delta_norm'] = _coalesce_columns(df, ['raw_delta_norm'])
    summary['filtered_delta_norm'] = _coalesce_columns(df, ['filtered_delta_norm'])
    summary['collapse_guard_triggered'] = _coalesce_columns(df, ['collapse_guard_triggered'])
    summary['rolled_back'] = _coalesce_columns(df, ['rolled_back'])
    return summary.dropna(axis=1, how='all')


def build_metrics_core(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    pre = df[df.get('stage', pd.Series(dtype=object)) == 'pre_attack_eval']
    warm = df[df.get('stage', pd.Series(dtype=object)) == 'warmup']
    attack = df[df.get('stage', pd.Series(dtype=object)).isin(['attack', 'cached_feature_attack'])]
    if len(pre):
        base_clean = float(pre.iloc[-1].get('clean_acc'))
    elif len(warm):
        base_clean = float(warm.iloc[-1].get('test_acc'))
    else:
        base_clean = float('nan')

    rows = []

    def add_row(name: str, row):
        clean_acc = row.get('clean_acc', row.get('test_acc', pd.NA))
        clean_drop = pd.NA
        if pd.notna(clean_acc) and pd.notna(base_clean):
            clean_drop = float(base_clean) - float(clean_acc)
        rows.append({
            'view': name,
            'stage': row.get('stage', pd.NA),
            'epoch': row.get('epoch', pd.NA),
            'clean_acc': clean_acc,
            'clean_drop_vs_pre': clean_drop,
            'input_asr_all': row.get('input_asr_all', pd.NA),
            'input_asr_non_target': row.get('input_asr_non_target', pd.NA),
            'latent_asr': row.get('latent_asr', pd.NA),
            'latent_masked_overwrite_asr': row.get('latent_masked_overwrite_asr', pd.NA),
            'latent_overwrite_asr': row.get('latent_overwrite_asr', pd.NA),
            'bridge_cos': row.get('bridge_cos', pd.NA),
            'bridge_rel_error': row.get('bridge_rel_error', pd.NA),
            'raw_delta_norm': row.get('raw_delta_norm', pd.NA),
            'filtered_delta_norm': row.get('filtered_delta_norm', pd.NA),
        })

    if len(warm):
        add_row('warmup_final', warm.iloc[-1])
    if len(pre):
        add_row('pre_attack_eval', pre.iloc[-1])
    if len(attack):
        add_row('attack_final', attack.iloc[-1])
        eligible = attack.copy()
        max_drop = 0.03
        if 'clean_acc' in eligible.columns and pd.notna(base_clean):
            eligible = eligible[eligible['clean_acc'] >= base_clean - max_drop]
        if len(eligible) and 'input_asr_non_target' in eligible.columns:
            add_row('best_attack_under_3pct_clean_drop', eligible.sort_values('input_asr_non_target', ascending=False).iloc[0])

    return pd.DataFrame(rows).dropna(axis=1, how='all')


def save_metrics_tables(rows, out_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    df.to_csv(os.path.join(out_dir, 'metrics_full.csv'), index=False)
    build_metrics_summary(df).to_csv(os.path.join(out_dir, 'metrics_summary.csv'), index=False)
    build_metrics_core(df).to_csv(os.path.join(out_dir, 'metrics_core.csv'), index=False)
    return df


def save_calibration(out_dir: str, calib, epoch_tag: str = 'epoch0') -> None:
    payload = {
        'trigger': calib.trigger.__dict__,
        'mean_shift_norm': calib.mean_shift_norm,
        'shift_std': calib.shift_std,
        'mahalanobis': calib.mahalanobis,
        'raw_delta_norm': float(getattr(calib, 'raw_delta_norm', 0.0)),
        'filtered_delta_norm': float(getattr(calib, 'filtered_delta_norm', 0.0)),
    }
    save_json(payload, os.path.join(out_dir, f'calibration_{epoch_tag}.json'))
    save_json(payload, os.path.join(out_dir, 'calibration.json'))


def remove_smashed_feature_cache(cache_dir: str, out_dir: str, reused: bool) -> bool:
    if reused or not os.path.isdir(cache_dir):
        return False
    cache_path = os.path.abspath(cache_dir)
    output_path = os.path.abspath(out_dir)
    if os.path.commonpath([cache_path, output_path]) != output_path:
        raise RuntimeError(f'Refusing to remove cache outside output_dir: {cache_path}')
    if os.path.basename(cache_path) != 'smashed_feature_cache':
        raise RuntimeError(f'Refusing to remove unexpected cache directory: {cache_path}')
    shutil.rmtree(cache_path)
    return True


def _to_uint8_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.nanmin(x))
    denom = float(np.nanmax(x))
    if denom > 1e-12:
        x = x / denom
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def save_delta_visualizations(out_dir: str, calib) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        save_json({'error': f'PIL import failed: {exc}'}, os.path.join(out_dir, 'delta_visualization_error.json'))
        return

    delta = calib.delta.detach().float().cpu()
    if delta.dim() == 4 and delta.shape[0] == 1:
        d = delta[0]
        spatial = torch.sqrt((d ** 2).sum(dim=0)).numpy()
        spatial_img = Image.fromarray(_to_uint8_image(spatial), mode='L').resize((256, 256), Image.Resampling.NEAREST)
        spatial_img.save(os.path.join(out_dir, 'delta_spatial_l2.png'))

        channel_norms = torch.sqrt((d ** 2).sum(dim=(1, 2)))
        topk = min(16, int(d.shape[0]))
        top_idx = torch.topk(channel_norms, k=topk).indices.tolist()
        tile = 96
        label_h = 18
        cols = 4
        rows = int(np.ceil(topk / cols))
        canvas = Image.new('RGB', (cols * tile, rows * (tile + label_h)), 'white')
        draw = ImageDraw.Draw(canvas)
        for rank, ci in enumerate(top_idx):
            arr = d[ci].numpy()
            arr = np.abs(arr)
            img = Image.fromarray(_to_uint8_image(arr), mode='L').resize((tile, tile), Image.Resampling.NEAREST).convert('RGB')
            x = (rank % cols) * tile
            y = (rank // cols) * (tile + label_h)
            canvas.paste(img, (x, y + label_h))
            draw.text((x + 3, y + 2), f'ch {ci}', fill=(0, 0, 0))
        canvas.save(os.path.join(out_dir, 'delta_top_channels_abs.png'))

        save_json({
            'delta_shape': list(delta.shape),
            'spatial_l2_png': 'delta_spatial_l2.png',
            'top_channels_abs_png': 'delta_top_channels_abs.png',
            'top_channels': [int(i) for i in top_idx],
            'top_channel_norms': [float(channel_norms[i].item()) for i in top_idx],
        }, os.path.join(out_dir, 'delta_visualization.json'))
    else:
        flat = delta.reshape(-1).numpy()
        np.save(os.path.join(out_dir, 'delta_flat.npy'), flat)
        save_json({
            'delta_shape': list(delta.shape),
            'note': 'Delta is not a 1xCxHxW feature map; saved flattened array only.',
        }, os.path.join(out_dir, 'delta_visualization.json'))


def print_epoch_summary(row: dict) -> None:
    def pct(key: str, default: float = 0.0) -> str:
        return f"{float(row.get(key, default)) * 100:.2f}%"

    def num(key: str, digits: int = 4, default: float = 0.0) -> str:
        return f"{float(row.get(key, default)):.{digits}f}"

    stage = row.get("stage", "unknown")
    epoch = int(row.get("epoch", 0))

    if stage in {"pre_attack_eval", "cached_feature_attack"}:
        stage_label = "Pre Attack" if stage == "pre_attack_eval" else "Cached Attack"
        clean_drop_pp = float(row.get("clean_drop_vs_pre", 0.0)) * 100
        clean_change_pp = 0.0 if abs(clean_drop_pp) < 0.005 else -clean_drop_pp
        bridge_cos = float(row.get("bridge_cos", 0.0))
        bridge_std = float(row.get("bridge_cos_std", 0.0))

        message = (
            f"[{stage_label}][Ep {epoch:02d}] "
            f"Clean {pct('clean_acc')} "
            f"({clean_change_pp:+.2f} pp) | "
            f"Input ASR {pct('input_asr_non_target')} | "
            f"Latent ASR {pct('latent_asr')} | "
            f"Target Latent {pct('target_latent_asr')} | "
        )
        if stage == "cached_feature_attack":
            poison_n = int(row.get("train_poison_n", row.get("poison_cached_count", 0)))
            cached_n = int(row.get("train_cached_n", row.get("target_cached_count", 0)))
            message += (
                f"Poison {poison_n}/{cached_n} "
                f"({pct('poison_fraction_actual')}) | "
                f"Loss {num('train_loss', 6)} | "
            )
        message += f"Bridge Cos {bridge_cos:.4f}±{bridge_std:.4f}"

    else:
        message = f"[{stage}][Ep {epoch:02d}] {row}"

    print(message, flush=True)




def main():
    parser = argparse.ArgumentParser(description='Split-learning backdoor experiment')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--clean-checkpoint', default=None, help='Clean split or full ResNet-18 checkpoint (.pt).')
    parser.add_argument('--device', default=None, help='cpu, cuda, or cuda:N')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--target-class', type=int, default=None)
    parser.add_argument('--split', choices=['early', 'middle', 'late'], default=None)
    parser.add_argument('--arch', choices=['smallcnn', 'resnet18', 'resnet34', 'mobilenetv2'], default=None)
    parser.add_argument('--trigger-name', default=None)
    parser.add_argument('--warmup-epochs', type=int, default=None)
    parser.add_argument('--attack-epochs', type=int, default=None)
    args = parser.parse_args()

    cfg = merge_override(load_config(args.config), args)
    set_seed(int(cfg.get('seed', 1)))
    device = get_device(cfg.get('device', 'cuda'))
    out_dir = cfg.get('output_dir', 'outputs/run')
    ensure_dir(out_dir)
    save_json(cfg, os.path.join(out_dir, 'resolved_config.json'))

    train_loader, test_loader, input_shape = get_dataloaders(cfg, seed=int(cfg.get('seed', 1)))
    client, server = build_split_model(cfg)
    client, server = client.to(device), server.to(device)

    opt_client = make_optimizer(client.parameters(), cfg, float(cfg['train']['lr_client']))
    opt_server = make_optimizer(server.parameters(), cfg, float(cfg['train']['lr_server']))

    rows = []
    cache_manifests = []
    cached_attack_enabled = bool(cfg['attack'].get('cached_feature_attack', {}).get('enabled', False))
    cache_epochs = int(cfg['attack'].get('cached_feature_attack', {}).get('cache_last_epochs', 2))
    cache_dir = os.path.join(out_dir, 'smashed_feature_cache')
    reuse_cache_dir = cfg['attack'].get('cached_feature_attack', {}).get('reuse_cache_dir')
    if cached_attack_enabled and not reuse_cache_dir and os.path.isdir(cache_dir):
        remove_smashed_feature_cache(cache_dir, out_dir, reused=False)

    warmup_epochs = int(cfg['train']['warmup_epochs'])
    ckpt_cfg = cfg.get('warmup_checkpoint', {})
    load_warmup_path = ckpt_cfg.get('load_path')
    if load_warmup_path:
        print(f'[1/5] Load clean warmup checkpoint: {load_warmup_path}')
        ckpt = load_clean_checkpoint(client, server, load_warmup_path, cfg)
        warm_eval = evaluate_clean(client, server, test_loader, device)
        row = {'stage': 'warmup_loaded', 'epoch': int(ckpt.get('epoch', warmup_epochs)), **{f'test_{k}': v for k, v in warm_eval.items()}}
        rows.append(row)
        print(row)
        if cached_attack_enabled and reuse_cache_dir:
            cache_dir = reuse_cache_dir
            files = sorted(
                os.path.join(cache_dir, name)
                for name in os.listdir(cache_dir)
                if name.endswith('.pt')
            )
            if not files:
                raise RuntimeError(f'reuse_cache_dir contains no .pt cache files: {cache_dir}')
            cache_manifests.append({'tag': 'reused_cache', 'files': files, 'num_samples': None})
            print(f'Reusing smashed-feature cache: {cache_dir} ({len(files)} files)')
        elif cached_attack_enabled:
            cache_dtype = cfg['attack'].get('cached_feature_attack', {}).get('cache_dtype', 'float32')
            manifest = cache_smashed_features(client, train_loader, device, cache_dir, tag='warmup_loaded', dtype=cache_dtype)
            cache_manifests.append(manifest)
    else:
        print('[1/4] Clean split-learning training')
        for ep in range(1, warmup_epochs + 1):
            tr = split_train_epoch_clean(client, server, train_loader, opt_client, opt_server, device)
            te = evaluate_clean(client, server, test_loader, device)
            row = {'stage': 'warmup', 'epoch': ep, **{f'train_{k}': v for k, v in tr.items()}, **{f'test_{k}': v for k, v in te.items()}}
            rows.append(row)
            print(row)
            if cached_attack_enabled and ep > warmup_epochs - cache_epochs:
                cache_dtype = cfg['attack'].get('cached_feature_attack', {}).get('cache_dtype', 'float32')
                manifest = cache_smashed_features(client, train_loader, device, cache_dir, tag=f'warmup_epoch{ep}', dtype=cache_dtype)
                cache_manifests.append(manifest)
    print('[2/4] Freeze client and open black-box smashed-feature queries')
    freeze_client_for_blackbox_queries(client)

    stats = collect_feature_stats(client, train_loader, device, max_batches=cfg['attack'].get('feature_stats_max_batches'))

    print('[3/4] Select delta from black-box carrier queries')
    calib = calibrate_input_latent_bridge(client, cfg, stats, input_shape, device)
    save_calibration(out_dir, calib)
    save_delta_visualizations(out_dir, calib)
    print('Trigger:', calib.trigger)

    print('[4/4] Backdoor training on the server')
    pre = {'stage': 'pre_attack_eval', 'epoch': 0, **eval_all(client, server, test_loader, calib, cfg, device)}
    pre.update(evaluate_target_class_latent_asr(client, server, test_loader, calib.delta, int(cfg['attack']['target_class']), device))
    rows.append(pre)
    print_epoch_summary(pre)
    baseline_clean_acc = float(pre['clean_acc'])

    opt_server = make_optimizer(server.parameters(), cfg, float(cfg['attack'].get('lr_server', cfg['train']['lr_server'])))

    if cached_attack_enabled:
        cfcfg = cfg['attack']['cached_feature_attack']
        cache_files = [path for manifest in cache_manifests for path in manifest['files']]
        mask_stats = assign_cached_poison_masks(
            cache_files,
            target_class=int(cfcfg.get('poison_class', cfg['attack']['target_class'])),
            poison_fraction=float(cfcfg.get('poison_fraction', 0.3)),
            seed=int(cfg.get('seed', 1)),
            source_classes=cfcfg.get('poison_classes'),
            label_mode=str(cfcfg.get('poison_label_mode', 'original')),
        )
        cache_info = {
            'cache_dir': cache_dir,
            'cache_files': len(cache_files),
            'cache_epochs': [m['tag'] for m in cache_manifests],
            **mask_stats,
        }
        save_json(cache_info, os.path.join(out_dir, 'cached_feature_attack.json'))

        best_score = -1e9
        best_epoch = 0
        for ep in range(1, int(cfg['train']['attack_epochs']) + 1):
            tr = train_server_epoch_from_cached_features(
                server,
                cache_files,
                opt_server,
                calib.delta,
                target_class=int(cfg['attack']['target_class']),
                cfg=cfg,
                device=device,
            )
            ev = eval_all(client, server, test_loader, calib, cfg, device)
            ev.update(evaluate_target_class_latent_asr(client, server, test_loader, calib.delta, int(cfg['attack']['target_class']), device))
            clean_drop = baseline_clean_acc - float(ev['clean_acc'])
            row = {
                'stage': 'cached_feature_attack',
                'epoch': ep,
                'clean_drop_vs_pre': clean_drop,
                **mask_stats,
                **{f'train_{k}': v for k, v in tr.items()},
                **ev,
            }
            score = float(ev.get('input_asr_non_target', 0.0)) + 0.5 * float(ev.get('latent_asr', 0.0)) - 4.0 * max(0.0, clean_drop)
            if score > best_score:
                best_score = score
                best_epoch = ep
            rows.append(row)
            print_epoch_summary(row)

        save_metrics_tables(rows, out_dir)
        cache_info['cache_removed'] = remove_smashed_feature_cache(cache_dir, out_dir, reused=bool(reuse_cache_dir))
        save_json(cache_info, os.path.join(out_dir, 'cached_feature_attack.json'))
        print(f'Done. Results saved to {out_dir}. Best epoch: {best_epoch}, score={best_score:.4f}')
        return

if __name__ == '__main__':
    main()

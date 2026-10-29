import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .carriers import make_carriers
from .datasets import get_dataloaders
from .models import build_split_model
from .triggers import TriggerSpec, apply_trigger
from .utils import ensure_dir, get_device, load_config, save_json, set_seed


@torch.no_grad()
def compute_raw_delta(client, cfg: Dict, input_shape, device: torch.device) -> Dict:
    client.eval()
    attack_cfg = cfg['attack']
    carrier_n = int(attack_cfg.get('carrier_count', 1024))
    carriers = make_carriers(carrier_n, input_shape, device=device, kind='mixed')
    trigger = TriggerSpec(
        name='checkerboard',
        patch_size=int(attack_cfg.get('patch_size', 4)),
        alpha=float(attack_cfg.get('trigger_alpha', 1.0)),
        position='center',
    )
    z0 = client(carriers).detach()
    zp = client(apply_trigger(carriers, trigger)).detach()
    shifts = zp - z0
    raw_delta = shifts.mean(dim=0, keepdim=True)
    centered = shifts - raw_delta
    cov_diag = centered.var(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return {
        'trigger': trigger,
        'raw_delta': raw_delta.detach().cpu(),
        'cov_diag': cov_diag.detach().cpu(),
        'carrier_count': carrier_n,
        'raw_delta_norm': float(raw_delta.norm().item()),
    }


def _epoch_cache_files(cache_dir: str, epoch_tag: str) -> List[str]:
    files = [
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.endswith('.pt') and name.startswith(epoch_tag)
    ]
    files = sorted(files)
    if not files:
        raise RuntimeError(f'No cache files for {epoch_tag} under {cache_dir}')
    return files


def collect_selected_channel_values(cache_files: List[str], channels: List[int]) -> torch.Tensor:
    chunks = []
    for path in cache_files:
        item = torch.load(path, map_location='cpu')
        z = item['z'].float()
        chunks.append(z[:, channels, :, :])
    return torch.cat(chunks, dim=0)


def build_position_table(raw_delta: torch.Tensor, cache_values: torch.Tensor, channels: List[int]) -> pd.DataFrame:
    rows = []
    d = raw_delta.squeeze(0).float()
    channel_norms = torch.sqrt((d ** 2).sum(dim=(1, 2)))
    _, h, w = d.shape
    for local_ci, channel in enumerate(channels):
        vals_ch = cache_values[:, local_ci, :, :]
        for hi in range(h):
            for wi in range(w):
                vals = vals_ch[:, hi, wi]
                rows.append({
                    'channel_rank': local_ci + 1,
                    'channel': int(channel),
                    'h': hi,
                    'w': wi,
                    'raw_delta_channel_l2_norm': float(channel_norms[channel].item()),
                    'smashed_max': float(vals.max().item()),
                    'smashed_min': float(vals.min().item()),
                    'smashed_mean': float(vals.mean().item()),
                    'smashed_median': float(vals.median().item()),
                    'raw_delta_value': float(d[channel, hi, wi].item()),
                    'sample_count': int(vals.numel()),
                })
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, path: str) -> None:
    columns = list(df.columns)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(columns) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for row in df.itertuples(index=False, name=None):
            f.write('| ' + ' | '.join(str(v) for v in row) + ' |\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--epoch-tag', default='warmup_epoch30')
    parser.add_argument('--top-channels', type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg['output_dir'] = args.output_dir
    cfg['attack']['trigger_name'] = 'checkerboard'
    cfg['attack']['trigger_position'] = 'center'
    cfg['attack']['patch_size'] = 4
    cfg['attack']['trigger_alpha'] = 1.0

    set_seed(int(cfg.get('seed', 1)))
    device = get_device(cfg.get('device', 'cuda'))
    ensure_dir(args.output_dir)
    save_json(cfg, os.path.join(args.output_dir, 'resolved_config.json'))

    _, _, input_shape = get_dataloaders(cfg, seed=int(cfg.get('seed', 1)))
    client, _ = build_split_model(cfg)
    client = client.to(device)
    ckpt_path = cfg.get('warmup_checkpoint', {}).get('load_path')
    if not ckpt_path:
        raise RuntimeError('warmup_checkpoint.load_path is required')
    ckpt = torch.load(ckpt_path, map_location=device)
    client.load_state_dict(ckpt['client'])
    client.eval()

    delta_payload = compute_raw_delta(client, cfg, input_shape, device)
    raw_delta = delta_payload['raw_delta']
    d = raw_delta.squeeze(0).float()
    channel_norms = torch.sqrt((d ** 2).sum(dim=(1, 2)))
    top_idx = torch.topk(channel_norms, k=min(args.top_channels, d.shape[0])).indices.tolist()

    np.savez_compressed(
        os.path.join(args.output_dir, 'raw_delta_uncut.npz'),
        raw_delta=raw_delta.numpy(),
        cov_diag=delta_payload['cov_diag'].numpy(),
        top_channels=np.asarray(top_idx, dtype=np.int64),
        top_channel_l2_norms=np.asarray([float(channel_norms[i].item()) for i in top_idx], dtype=np.float32),
    )

    cache_files = _epoch_cache_files(args.cache_dir, args.epoch_tag)
    cache_values = collect_selected_channel_values(cache_files, top_idx)
    table = build_position_table(raw_delta, cache_values, top_idx)
    table.to_csv(os.path.join(args.output_dir, 'epoch30_top8_raw_delta_per_position_stats.csv'), index=False)
    write_markdown_table(table, os.path.join(args.output_dir, 'epoch30_top8_raw_delta_per_position_stats.md'))

    save_json({
        'cache_dir': args.cache_dir,
        'epoch_tag': args.epoch_tag,
        'cache_files': len(cache_files),
        'sample_count': int(cache_values.shape[0]),
        'raw_delta_shape': list(raw_delta.shape),
        'raw_delta_norm': delta_payload['raw_delta_norm'],
        'trigger': delta_payload['trigger'].__dict__,
        'top_channels': [int(i) for i in top_idx],
        'top_channel_l2_norms': [float(channel_norms[i].item()) for i in top_idx],
        'outputs': {
            'raw_delta': 'raw_delta_uncut.npz',
            'table_csv': 'epoch30_top8_raw_delta_per_position_stats.csv',
            'table_md': 'epoch30_top8_raw_delta_per_position_stats.md',
        },
    }, os.path.join(args.output_dir, 'raw_delta_position_stats_summary.json'))


if __name__ == '__main__':
    main()

import argparse
import os

import torch

from .attack import collect_feature_stats
from .datasets import get_dataloaders
from .models import build_split_model
from .run_experiment import freeze_client_for_blackbox_queries
from .run_zp_topk_masked_overwrite_experiment import build_zp_topk_delta
from .triggers import apply_trigger
from .utils import get_device, load_config, save_json, set_seed


@torch.no_grad()
def evaluate_zp_delta_cos(client, loader, trigger, delta, device):
    client.eval()
    d = delta.to(device)
    mask = d != 0
    df = d.reshape(1, -1)
    full_cos_values = []
    masked_cos_values = []
    zp_norm_values = []
    masked_zp_norm_values = []

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        zp = client(apply_trigger(x, trigger))
        zpf = zp.reshape(zp.shape[0], -1)
        full_cos = (zpf * df).sum(dim=1) / (zpf.norm(dim=1) * df.norm(dim=1) + 1e-8)
        full_cos_values.append(full_cos.cpu())
        zp_norm_values.append(zpf.norm(dim=1).cpu())

        masked_zp = zp * mask
        mzpf = masked_zp.reshape(masked_zp.shape[0], -1)
        masked_cos = (mzpf * df).sum(dim=1) / (mzpf.norm(dim=1) * df.norm(dim=1) + 1e-8)
        masked_cos_values.append(masked_cos.cpu())
        masked_zp_norm_values.append(mzpf.norm(dim=1).cpu())

    full = torch.cat(full_cos_values)
    masked = torch.cat(masked_cos_values)
    zp_norm = torch.cat(zp_norm_values)
    masked_zp_norm = torch.cat(masked_zp_norm_values)
    return {
        'zp_delta_cos': float(full.mean().item()),
        'zp_delta_cos_std': float(full.std(unbiased=False).item()),
        'zp_masked_delta_cos': float(masked.mean().item()),
        'zp_masked_delta_cos_std': float(masked.std(unbiased=False).item()),
        'zp_norm': float(zp_norm.mean().item()),
        'zp_masked_norm': float(masked_zp_norm.mean().item()),
        'delta_norm': float(d.norm().item()),
        'delta_nonzero': int(mask.sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--trigger-name', required=True)
    parser.add_argument('--output-json', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg['attack']['trigger_name'] = args.trigger_name

    set_seed(int(cfg.get('seed', 1)))
    device = get_device(cfg.get('device', 'cuda'))
    train_loader, test_loader, input_shape = get_dataloaders(cfg, seed=int(cfg.get('seed', 1)))
    client, server = build_split_model(cfg)
    client, server = client.to(device), server.to(device)

    ckpt_path = cfg.get('warmup_checkpoint', {}).get('load_path')
    if not ckpt_path:
        raise RuntimeError('This script expects warmup_checkpoint.load_path to be set.')
    ckpt = torch.load(ckpt_path, map_location=device)
    client.load_state_dict(ckpt['client'])
    server.load_state_dict(ckpt['server'])
    freeze_client_for_blackbox_queries(client)

    stats = collect_feature_stats(client, train_loader, device, max_batches=cfg['attack'].get('feature_stats_max_batches'))
    calib = build_zp_topk_delta(client, cfg, stats, input_shape, device)
    metrics = evaluate_zp_delta_cos(client, test_loader, calib.trigger, calib.delta, device)
    metrics.update({
        'trigger': calib.trigger.__dict__,
        'raw_delta_norm': float(calib.raw_delta_norm),
        'filtered_delta_norm': float(calib.filtered_delta_norm),
        'delta_topk_ratio': float(cfg['attack'].get('delta_topk_ratio', 0.14)),
        'carrier_count': int(cfg['attack'].get('carrier_count', 16)),
    })
    print(metrics)

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_json(metrics, args.output_json)


if __name__ == '__main__':
    main()

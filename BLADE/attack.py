from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .carriers import make_carriers
from .triggers import TriggerSpec, apply_trigger
from .utils import flatten_batch


@dataclass
class FeatureStats:
    mean: torch.Tensor
    var: torch.Tensor
    shape: Tuple[int, ...]


@dataclass
class CalibrationResult:
    trigger: TriggerSpec
    delta: torch.Tensor
    cov_diag: torch.Tensor
    mean_shift_norm: float
    shift_std: float
    mahalanobis: float
    snr_mask: Optional[torch.Tensor] = None
    raw_delta_norm: float = 0.0
    filtered_delta_norm: float = 0.0
    shift_bank: Optional[torch.Tensor] = None


@torch.no_grad()
def collect_feature_stats(client: nn.Module, loader: Iterable, device: torch.device, max_batches: Optional[int] = None) -> FeatureStats:
    client.eval()
    total_n = 0
    sum_z = None
    sumsq_z = None
    shape = None
    for bi, (x, _) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        z = client(x)
        if shape is None:
            shape = tuple(z.shape[1:])
        zflat = flatten_batch(z).detach().float().cpu()
        batch_sum = zflat.sum(dim=0)
        batch_sumsq = (zflat * zflat).sum(dim=0)
        if sum_z is None:
            sum_z = batch_sum
            sumsq_z = batch_sumsq
        else:
            sum_z += batch_sum
            sumsq_z += batch_sumsq
        total_n += int(zflat.shape[0])
    if total_n == 0 or sum_z is None or sumsq_z is None or shape is None:
        raise RuntimeError('No features were collected for statistics.')
    mean = sum_z / float(total_n)
    var = (sumsq_z / float(total_n) - mean * mean).clamp_min(1e-6)
    return FeatureStats(mean=mean, var=var, shape=shape)


def _mahalanobis_diag(v: torch.Tensor, stats: FeatureStats) -> torch.Tensor:
    vf = v.reshape(-1).detach().cpu()
    return ((vf ** 2) / stats.var).mean().sqrt()


def _apply_delta_snr_filter(delta: torch.Tensor, cov_diag: torch.Tensor, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    raw_norm = float(delta.norm().item())
    ratio = float(cfg['attack'].get('delta_topk_ratio', 1.0))
    ratio = max(0.0, min(1.0, ratio))
    flat_delta = delta.reshape(-1)
    flat_var = cov_diag.reshape(-1).clamp_min(1e-8)
    if ratio < 1.0:
        k = max(1, int(flat_delta.numel() * ratio))
        snr = flat_delta.abs() / flat_var.sqrt()
        idx = torch.topk(snr, k=k, largest=True).indices
        mask_flat = torch.zeros_like(flat_delta)
        mask_flat[idx] = 1.0
        mask = mask_flat.reshape_as(delta)
        delta = delta * mask
        cov_diag = cov_diag * mask + cov_diag.mean().detach() * (1.0 - mask)
    else:
        mask = torch.ones_like(delta)

    scale = float(cfg['attack'].get('delta_scale', 1.0))
    delta = delta * scale
    norm_clip = cfg['attack'].get('delta_norm_clip', None)
    if norm_clip is not None:
        norm_clip = float(norm_clip)
        dn = float(delta.norm().item())
        if dn > norm_clip:
            delta = delta * (norm_clip / (dn + 1e-8))
    filtered_norm = float(delta.norm().item())
    return delta, mask, raw_norm, filtered_norm


def _apply_zp_topk_filter(
    mean_zp: torch.Tensor,
    stats: FeatureStats,
    cfg: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    shape = tuple(mean_zp.shape[1:])
    normal_mean = stats.mean.reshape(1, *shape).to(device)
    normal_var = stats.var.reshape(1, *shape).to(device).clamp_min(1e-8)
    ratio = max(0.0, min(1.0, float(cfg['attack'].get('delta_topk_ratio', 1.0))))
    flat_zp = mean_zp.reshape(-1)
    snr = (mean_zp - normal_mean).abs().reshape(-1) / normal_var.sqrt().reshape(-1)
    if ratio < 1.0:
        k = max(1, int(flat_zp.numel() * ratio))
        idx = torch.topk(snr, k=k, largest=True).indices
        mask_flat = torch.zeros_like(flat_zp)
        mask_flat[idx] = 1.0
        mask = mask_flat.reshape_as(mean_zp)
        delta = mean_zp * mask
    else:
        mask = torch.ones_like(mean_zp)
        delta = mean_zp
    cov_diag = normal_var * mask + normal_var.mean().detach() * (1.0 - mask)
    raw_norm = float(mean_zp.norm().item())

    scale = float(cfg['attack'].get('delta_scale', 1.0))
    delta = delta * scale
    norm_clip = cfg['attack'].get('delta_norm_clip', None)
    if norm_clip is not None:
        norm_clip = float(norm_clip)
        dn = float(delta.norm().item())
        if dn > norm_clip:
            delta = delta * (norm_clip / (dn + 1e-8))
    filtered_norm = float(delta.norm().item())
    return delta, mask, cov_diag, raw_norm, filtered_norm


@torch.no_grad()
def calibrate_input_latent_bridge(
    client: nn.Module,
    cfg: dict,
    stats: FeatureStats,
    input_shape: Tuple[int, int, int],
    device: torch.device,
) -> CalibrationResult:
    client.eval()
    carrier_n = int(cfg['attack'].get('carrier_count', 256))
    carriers = make_carriers(carrier_n, input_shape, device=device, kind='mixed')
    trigger_name = str(cfg['attack'].get('trigger_name', 'checkerboard'))
    patch_size = int(cfg['attack'].get('patch_size', 4))
    alpha = float(cfg['attack'].get('trigger_alpha', 1.0))
    position = str(cfg['attack'].get('trigger_position', 'bottom_right'))
    color_a = cfg['attack'].get('trigger_color_a')
    color_b = cfg['attack'].get('trigger_color_b')
    delta_source = str(cfg['attack'].get('delta_source', 'shift')).lower()
    z0 = client(carriers).detach()
    spec = TriggerSpec(
        name=trigger_name,
        patch_size=patch_size,
        alpha=alpha,
        position=position,
        color_a=color_a,
        color_b=color_b,
    )
    zp = client(apply_trigger(carriers, spec)).detach()
    if delta_source in {'zp', 'mean_zp', 'triggered_z'}:
        mean_zp = zp.mean(dim=0, keepdim=True)
        delta, mask, cov_diag, raw_norm, filtered_norm = _apply_zp_topk_filter(
            mean_zp.detach(), stats, cfg, device
        )
        filtered_shifts = zp * mask.to(zp.device)
        shift_bank = None
    else:
        shifts = zp - z0
        mean_shift = shifts.mean(dim=0, keepdim=True)
        centered = flatten_batch(shifts - mean_shift)
        cov_diag = centered.var(dim=0, unbiased=False).reshape_as(mean_shift).clamp_min(1e-6)
        delta, mask, raw_norm, filtered_norm = _apply_delta_snr_filter(
            mean_shift.detach(), cov_diag.detach(), cfg
        )
        filtered_shifts = shifts * mask.to(shifts.device)
        shift_bank = filtered_shifts.detach()

    mean_norm = float(delta.norm().item())
    shift_std = float(flatten_batch(filtered_shifts - delta).norm(dim=1).mean().item() + 1e-8)
    mahalanobis = float(_mahalanobis_diag(delta, stats).item())
    if not bool(cfg['attack'].get('use_calibrated_magnitude', False)):
        delta = delta / (delta.norm() + 1e-8)
    return CalibrationResult(
        trigger=spec,
        delta=delta.to(device),
        cov_diag=cov_diag.to(device),
        mean_shift_norm=mean_norm,
        shift_std=shift_std,
        mahalanobis=mahalanobis,
        snr_mask=mask.to(device),
        raw_delta_norm=float(raw_norm),
        filtered_delta_norm=float(filtered_norm),
        shift_bank=None if shift_bank is None else shift_bank.to(device),
    )


def make_latent_triggered_z(
    z: torch.Tensor,
    delta: torch.Tensor,
    cov_diag: Optional[torch.Tensor],
    cfg: dict,
    shift_bank: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    amin = float(cfg['attack'].get('latent_scale_min', 0.5))
    amax = float(cfg['attack'].get('latent_scale_max', 1.5))
    noise_std = float(cfg['attack'].get('latent_noise_std', 0.03))
    b = z.shape[0]
    scales = torch.empty(b, *([1] * (z.dim() - 1)), device=z.device).uniform_(amin, amax)

    if shift_bank is not None and bool(cfg['attack'].get('sample_carrier_shifts', False)):
        bank = shift_bank.to(z.device)
        idx = torch.randint(0, bank.shape[0], (b,), device=z.device)
        base_delta = bank[idx]
    else:
        base_delta = delta.to(z.device)

    noise = torch.randn_like(z) * noise_std
    if cov_diag is not None:
        noise = noise * cov_diag.sqrt().to(z.device)
    return z + scales * base_delta + noise


def margin_loss(logits: torch.Tensor, target_class: int, margin: float) -> torch.Tensor:
    yt = logits[:, target_class]
    masked = logits.clone()
    masked[:, target_class] = -1e9
    other = masked.max(dim=1).values
    return F.relu(margin - (yt - other)).mean()

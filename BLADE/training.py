import math
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .attack import (
    make_latent_triggered_z,
    margin_loss,
)
from .triggers import apply_trigger
from .utils import top1_accuracy


def split_train_epoch_clean(
    client: nn.Module,
    server: nn.Module,
    loader,
    opt_client: torch.optim.Optimizer,
    opt_server: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    client.train(); server.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc='clean-train', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt_client.zero_grad(set_to_none=True)
        opt_server.zero_grad(set_to_none=True)

        z_client = client(x)
        z_server = z_client.detach().requires_grad_(True)
        logits = server(z_server)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad_z = z_server.grad.detach()
        opt_server.step()

        z_client.backward(grad_z)
        opt_client.step()

        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_acc += top1_accuracy(logits.detach(), y) * bs
        total_n += bs
    return {'loss': total_loss / total_n, 'acc': total_acc / total_n}


@torch.no_grad()
def cache_smashed_features(
    client: nn.Module,
    loader,
    device: torch.device,
    cache_dir: str,
    tag: str,
    dtype: str = 'float32',
) -> Dict[str, object]:
    client.eval()
    os.makedirs(cache_dir, exist_ok=True)
    dtype = str(dtype).lower()
    if dtype not in {'float32', 'float16'}:
        raise ValueError(f'Unsupported smashed feature cache dtype: {dtype}')
    files: List[str] = []
    total_n = 0
    for bi, (x, y) in enumerate(tqdm(loader, desc=f'cache-{tag}', leave=False)):
        x = x.to(device, non_blocking=True)
        z = client(x).detach().cpu()
        if dtype == 'float16':
            z = z.half()
        y_cpu = y.detach().cpu().long()
        path = os.path.join(cache_dir, f'{tag}_batch{bi:05d}.pt')
        torch.save({'z': z, 'y': y_cpu}, path)
        files.append(path)
        total_n += int(y_cpu.numel())
    return {'tag': tag, 'files': files, 'num_samples': total_n}


def assign_cached_poison_masks(
    cache_files: List[str],
    target_class: int,
    poison_fraction: float,
    seed: int,
    source_classes: Optional[Union[List[int], str]] = None,
    label_mode: str = 'original',
) -> Dict[str, float]:
    label_mode = str(label_mode).lower()
    if label_mode not in {'original', 'target'}:
        raise ValueError(f'Unsupported cached_feature_attack.poison_label_mode: {label_mode}')
    discover_all_classes = isinstance(source_classes, str) and source_classes.lower() == 'all'
    if source_classes is None:
        source_classes = [int(target_class)]
    elif discover_all_classes:
        discovered = set()
        for path in cache_files:
            item = torch.load(path, map_location='cpu')
            discovered.update(int(y) for y in item['y'].unique().tolist())
        source_classes = sorted(discovered)
    elif isinstance(source_classes, str):
        raise ValueError(f'Unsupported cached_feature_attack.poison_classes: {source_classes}')
    source_classes = [int(c) for c in source_classes]

    positions_by_class: Dict[int, List[tuple]] = {cls: [] for cls in source_classes}
    for fi, path in enumerate(cache_files):
        item = torch.load(path, map_location='cpu')
        ys = item['y']
        for cls in source_classes:
            cls_pos = torch.nonzero(ys == cls, as_tuple=False).flatten().tolist()
            positions_by_class[cls].extend((fi, int(pos)) for pos in cls_pos)

    g = torch.Generator().manual_seed(int(seed))
    selected_positions = set()
    selected_counts: Dict[int, int] = {}
    eligible_count = 0
    for cls, positions in positions_by_class.items():
        eligible_count += len(positions)
        poison_count_cls = int(round(len(positions) * float(poison_fraction)))
        selected_counts[cls] = poison_count_cls
        if poison_count_cls <= 0:
            continue
        selected = torch.randperm(len(positions), generator=g)[:poison_count_cls].tolist()
        selected_positions.update(positions[i] for i in selected)
    poison_count = len(selected_positions)

    by_file: Dict[int, List[int]] = {}
    for fi, pos in selected_positions:
        by_file.setdefault(fi, []).append(pos)

    for fi, path in enumerate(cache_files):
        item = torch.load(path, map_location='cpu')
        mask = torch.zeros_like(item['y'], dtype=torch.bool)
        if fi in by_file:
            mask[torch.tensor(by_file[fi], dtype=torch.long)] = True
        item['poison_mask'] = mask
        if label_mode == 'target':
            poison_target_y = item['y'].clone()
            poison_target_y[mask] = int(target_class)
            item['poison_target_y'] = poison_target_y
        elif 'poison_target_y' in item:
            item.pop('poison_target_y')
        torch.save(item, path)

    return {
        'target_cached_count': float(eligible_count),
        'poison_cached_count': float(poison_count),
        'poison_fraction_actual': float(poison_count / max(1, eligible_count)),
        'poison_source_class_count': float(len(source_classes)),
        'poison_label_mode_target': float(label_mode == 'target'),
        **{f'poison_cached_count_class_{cls}': float(selected_counts.get(cls, 0)) for cls in source_classes},
    }


def train_server_epoch_from_cached_features(
    server: nn.Module,
    cache_files: List[str],
    opt_server: torch.optim.Optimizer,
    delta: torch.Tensor,
    target_class: int,
    cfg: dict,
    device: torch.device,
) -> Dict[str, float]:
    server.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    poison_loss, poison_acc, poison_n = 0.0, 0.0, 0
    clean_loss, clean_acc, clean_n = 0.0, 0.0, 0
    d = delta.to(device)
    cached_attack_cfg = cfg['attack'].get('cached_feature_attack', {})
    poison_mode = str(cached_attack_cfg.get('poison_mode', 'add')).lower()
    if poison_mode not in {'add', 'overwrite', 'masked_overwrite'}:
        raise ValueError(f'Unsupported cached_feature_attack.poison_mode: {poison_mode}')
    poison_noise_mix = cached_attack_cfg.get('poison_noise_mix')
    poison_noise_std = float(cached_attack_cfg.get('poison_noise_std', 0.0))
    if poison_noise_std < 0:
        raise ValueError(f'cached_feature_attack.poison_noise_std must be non-negative: {poison_noise_std}')
    poison_noise_type = str(cached_attack_cfg.get('poison_noise_type', 'gaussian')).lower()
    valid_noise_types = {'gaussian', 'uniform', 'rademacher'}
    if poison_noise_type not in valid_noise_types:
        raise ValueError(f'Unsupported cached_feature_attack.poison_noise_type: {poison_noise_type}')
    if poison_noise_mix:
        for component in poison_noise_mix:
            component_type = str(component.get('type', 'gaussian')).lower()
            component_std = float(component.get('std', 0.0))
            if component_type not in valid_noise_types:
                raise ValueError(f'Unsupported cached_feature_attack.poison_noise_mix type: {component_type}')
            if component_std < 0:
                raise ValueError(f'cached_feature_attack.poison_noise_mix std must be non-negative: {component_std}')

    def make_single_noise_like(x: torch.Tensor, noise_type: str, noise_std: float) -> torch.Tensor:
        if noise_std == 0:
            return torch.zeros_like(x)
        if noise_type == 'gaussian':
            return torch.randn_like(x) * noise_std
        if noise_type == 'uniform':
            bound = math.sqrt(3.0) * noise_std
            return (torch.rand_like(x) * 2.0 - 1.0) * bound
        return torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0).mul_(noise_std)

    def make_poison_noise_like(x: torch.Tensor) -> torch.Tensor:
        if not poison_noise_mix:
            return make_single_noise_like(x, poison_noise_type, poison_noise_std)

        noise = torch.zeros_like(x)
        sample_count = x.shape[0]
        component_idx = torch.randint(len(poison_noise_mix), (sample_count,), device=x.device)
        for ci, component in enumerate(poison_noise_mix):
            sample_mask = component_idx == ci
            if not bool(sample_mask.any()):
                continue
            component_type = str(component.get('type', 'gaussian')).lower()
            component_std = float(component.get('std', 0.0))
            noise[sample_mask] = make_single_noise_like(x[sample_mask], component_type, component_std)
        return noise

    for path in tqdm(cache_files, desc='cached-server-train', leave=False):
        item = torch.load(path, map_location='cpu')
        z = item['z'].to(device, non_blocking=True).float()
        y = item.get('poison_target_y', item['y']).to(device, non_blocking=True)
        poison_mask = item.get('poison_mask', torch.zeros_like(item['y'], dtype=torch.bool)).to(device, non_blocking=True)

        z_train = z.clone()
        if bool(poison_mask.any()):
            if poison_mode == 'overwrite':
                poison_z = d.expand_as(z_train[poison_mask])
                if poison_noise_std > 0:
                    poison_z = poison_z + make_poison_noise_like(poison_z)
                z_train[poison_mask] = poison_z
            elif poison_mode == 'masked_overwrite':
                poison_z = z_train[poison_mask].clone()
                overwrite_mask = d.expand_as(poison_z) != 0
                poison_values = d.expand_as(poison_z)
                if poison_noise_std > 0:
                    poison_values = poison_values + make_poison_noise_like(poison_values)
                poison_z[overwrite_mask] = poison_values[overwrite_mask]
                z_train[poison_mask] = poison_z
            else:
                poison_delta = d
                if poison_noise_std > 0:
                    poison_delta = poison_delta + make_poison_noise_like(z_train[poison_mask])
                z_train[poison_mask] = z_train[poison_mask] + poison_delta

        opt_server.zero_grad(set_to_none=True)
        logits = server(z_train)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt_server.step()

        bs = int(y.numel())
        acc = top1_accuracy(logits.detach(), y)
        total_loss += float(loss.item()) * bs
        total_acc += acc * bs
        total_n += bs

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            if bool(poison_mask.any()):
                pn = int(poison_mask.sum().item())
                poison_loss += F.cross_entropy(logits[poison_mask], y[poison_mask]).item() * pn
                poison_acc += (pred[poison_mask] == y[poison_mask]).float().sum().item()
                poison_n += pn
            clean_mask = ~poison_mask
            if bool(clean_mask.any()):
                cn = int(clean_mask.sum().item())
                clean_loss += F.cross_entropy(logits[clean_mask], y[clean_mask]).item() * cn
                clean_acc += (pred[clean_mask] == y[clean_mask]).float().sum().item()
                clean_n += cn

    return {
        'loss': total_loss / max(1, total_n),
        'acc': total_acc / max(1, total_n),
        'poison_loss': poison_loss / max(1, poison_n),
        'poison_acc': poison_acc / max(1, poison_n),
        'clean_cached_loss': clean_loss / max(1, clean_n),
        'clean_cached_acc': clean_acc / max(1, clean_n),
        'poison_n': float(poison_n),
        'cached_n': float(total_n),
        'poison_mode_overwrite': float(poison_mode == 'overwrite'),
        'poison_mode_masked_overwrite': float(poison_mode == 'masked_overwrite'),
        'poison_noise_std': poison_noise_std,
        'poison_noise_type_' + poison_noise_type: 1.0,
        'poison_noise_mix': float(bool(poison_noise_mix)),
    }


@torch.no_grad()
def evaluate_target_class_latent_asr(
    client: nn.Module,
    server: nn.Module,
    loader,
    delta: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> Dict[str, float]:
    client.eval()
    server.eval()
    total, success = 0, 0.0
    d = delta.to(device)
    for x, y in loader:
        mask = y == target_class
        if not bool(mask.any()):
            continue
        x = x[mask].to(device, non_blocking=True)
        z = client(x) + d
        pred = server(z).argmax(dim=1)
        success += (pred == target_class).float().sum().item()
        total += int(mask.sum().item())
    return {'target_latent_asr': success / max(1, total), 'target_latent_n': float(total)}


@torch.no_grad()
def evaluate_latent_overwrite_attack(
    client: nn.Module,
    server: nn.Module,
    loader,
    delta: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> Dict[str, float]:
    client.eval()
    server.eval()
    total, success = 0, 0.0
    d = delta.to(device)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z = client(x)
        zt = d.expand_as(z)
        pred = server(zt).argmax(dim=1)
        success += (pred == target_class).float().sum().item()
        total += x.shape[0]
    return {'latent_overwrite_asr': success / max(1, total)}


@torch.no_grad()
def evaluate_latent_masked_overwrite_attack(
    client: nn.Module,
    server: nn.Module,
    loader,
    delta: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> Dict[str, float]:
    client.eval()
    server.eval()
    total, success = 0, 0.0
    d = delta.to(device)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z = client(x)
        zt = z.clone()
        expanded = d.expand_as(zt)
        mask = expanded != 0
        zt[mask] = expanded[mask]
        pred = server(zt).argmax(dim=1)
        success += (pred == target_class).float().sum().item()
        total += x.shape[0]
    return {'latent_masked_overwrite_asr': success / max(1, total)}


def split_train_epoch_backdoor(
    client: nn.Module,
    server: nn.Module,
    loader,
    opt_server: torch.optim.Optimizer,
    delta: torch.Tensor,
    cov_diag: Optional[torch.Tensor],
    cfg: dict,
    device: torch.device,
    shift_bank: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    client.eval()
    for p in client.parameters():
        p.requires_grad_(False)
    server.train()

    target = int(cfg['attack']['target_class'])
    lam_bd = float(cfg['attack'].get('lambda_bd', 0.1))
    lam_clean = float(cfg['attack'].get('lambda_clean', 1.0))
    lam_cons = float(cfg['attack'].get('lambda_cons', 0.0))
    lam_margin = float(cfg['attack'].get('lambda_margin', 0.0))
    margin = float(cfg['attack'].get('margin', 1.0))

    totals = {
        'loss': 0.0, 'clean_loss': 0.0, 'bd_loss': 0.0, 'cons_loss': 0.0,
        'margin_loss': 0.0, 'acc': 0.0, 'latent_asr': 0.0, 'n': 0
    }

    for x, y in tqdm(loader, desc='backdoor-train', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.shape[0]
        target_y = torch.full_like(y, target)

        opt_server.zero_grad(set_to_none=True)

        with torch.no_grad():
            z_server = client(x).detach()

        z_bd = make_latent_triggered_z(z_server, delta, cov_diag, cfg, shift_bank=shift_bank)
        logits_clean = server(z_server)
        logits_bd = server(z_bd)

        clean_loss = F.cross_entropy(logits_clean, y)
        bd_loss = F.cross_entropy(logits_bd, target_y)
        cons_loss = logits_clean.new_tensor(0.0)
        m_loss = margin_loss(logits_bd, target, margin)
        loss_server = (
            lam_clean * clean_loss
            + lam_bd * bd_loss
            + lam_cons * cons_loss
            + lam_margin * m_loss
        )

        loss_server.backward()
        opt_server.step()

        with torch.no_grad():
            totals['loss'] += loss_server.item() * bs
            totals['clean_loss'] += clean_loss.item() * bs
            totals['bd_loss'] += bd_loss.item() * bs
            totals['cons_loss'] += cons_loss.item() * bs
            totals['margin_loss'] += m_loss.item() * bs
            totals['acc'] += top1_accuracy(logits_clean.detach(), y) * bs
            totals['latent_asr'] += (logits_bd.argmax(dim=1) == target_y).float().mean().item() * bs
            totals['n'] += bs

    n = totals.pop('n')
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate_clean(
    client: nn.Module,
    server: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    client.eval(); server.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = client(x)
        logits = server(z)
        loss = F.cross_entropy(logits, y)
        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_acc += top1_accuracy(logits, y) * bs
        total_n += bs
    return {'loss': total_loss / total_n, 'acc': total_acc / total_n}


@torch.no_grad()
def evaluate_input_attack(
    client, server, loader, trigger, target_class: int, device: torch.device,
) -> Dict[str, float]:
    client.eval(); server.eval()
    total_asr, total_n = 0.0, 0
    non_target_asr, non_target_n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        xt = apply_trigger(x, trigger)
        z = client(xt)
        logits = server(z)
        pred = logits.argmax(dim=1)
        target_y = torch.full_like(y, target_class)
        bs = x.shape[0]
        total_asr += (pred == target_y).float().sum().item()
        total_n += bs
        mask = y != target_class
        if bool(mask.any()):
            non_target_asr += (pred[mask] == target_class).float().sum().item()
            non_target_n += int(mask.sum().item())
    return {
        'input_asr_all': total_asr / total_n,
        'input_asr_non_target': non_target_asr / max(1, non_target_n),
    }


@torch.no_grad()
def evaluate_latent_attack(
    client, server, loader, delta, cov_diag, target_class: int, cfg: dict, device: torch.device,
    shift_bank: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    client.eval(); server.eval()
    total_asr, total_n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = client(x)
        zt = make_latent_triggered_z(z, delta, cov_diag, cfg, shift_bank=shift_bank)
        logits = server(zt)
        pred = logits.argmax(dim=1)
        total_asr += (pred == target_class).float().sum().item()
        total_n += x.shape[0]
    return {'latent_asr': total_asr / total_n}


@torch.no_grad()
def evaluate_bridge(client, loader, trigger, delta, device: torch.device) -> Dict[str, float]:
    client.eval()
    cos_values, err_values, norm_values = [], [], []
    d = delta.to(device)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z0 = client(x)
        zp = client(apply_trigger(x, trigger))
        shift = zp - z0
        sf = shift.reshape(shift.shape[0], -1)
        df = d.reshape(1, -1)
        cos = (sf * df).sum(dim=1) / (sf.norm(dim=1) * df.norm(dim=1) + 1e-8)
        proj = (sf * df).sum(dim=1, keepdim=True) / (df.norm(dim=1, keepdim=True) ** 2 + 1e-8) * df
        err = (sf - proj).norm(dim=1) / (sf.norm(dim=1) + 1e-8)
        cos_values.append(cos.cpu())
        err_values.append(err.cpu())
        norm_values.append(sf.norm(dim=1).cpu())
    cos_v = torch.cat(cos_values)
    err_v = torch.cat(err_values)
    norm_v = torch.cat(norm_values)
    return {
        'bridge_cos': float(cos_v.mean().item()),
        'bridge_cos_std': float(cos_v.std(unbiased=False).item()),
        'bridge_rel_error': float(err_v.mean().item()),
        'trigger_shift_norm': float(norm_v.mean().item()),
        'delta_norm_eval': float(d.norm().item()),
    }

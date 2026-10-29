import argparse
import copy
import json
import os
from dataclasses import dataclass, asdict, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .datasets import NpzImageDataset, TinyImageNetVal
from .models import build_split_model, _make_resnet_layer
from .utils import ensure_dir, get_device, set_seed, top1_accuracy


@dataclass
class CleanJob:
    name: str
    dataset_name: str
    data_dir: str
    in_channels: int
    num_classes: int
    input_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    momentum: float
    splits: Sequence[str]
    npz_path: Optional[str] = None
    num_workers: int = 2


DEFAULT_JOBS: Dict[str, CleanJob] = {
    'cifar10': CleanJob(
        name='cifar10',
        dataset_name='cifar10',
        data_dir='./data',
        in_channels=3,
        num_classes=10,
        input_size=32,
        epochs=100,
        batch_size=512,
        lr=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        splits=('early', 'middle', 'late'),
    ),
    'cifar100': CleanJob(
        name='cifar100',
        dataset_name='cifar100',
        data_dir='./data',
        in_channels=3,
        num_classes=100,
        input_size=32,
        epochs=150,
        batch_size=512,
        lr=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        splits=('early', 'middle'),
    ),
    'pathmnist': CleanJob(
        name='pathmnist',
        dataset_name='pathmnist',
        data_dir='./data',
        npz_path='./data/pathmnist_64.npz',
        in_channels=3,
        num_classes=9,
        input_size=64,
        epochs=80,
        batch_size=256,
        lr=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        splits=('early', 'middle'),
    ),
    'gtsrb': CleanJob(
        name='gtsrb',
        dataset_name='gtsrb',
        data_dir='./data',
        in_channels=3,
        num_classes=43,
        input_size=32,
        epochs=80,
        batch_size=512,
        lr=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        splits=('early',),
    ),
    'tinyimagenet': CleanJob(
        name='tinyimagenet',
        dataset_name='tinyimagenet',
        data_dir='./data',
        in_channels=3,
        num_classes=200,
        input_size=64,
        epochs=240,
        batch_size=256,
        lr=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        splits=('early', 'middle'),
    ),
}


class FullResNet18(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1, c1 = _make_resnet_layer(64, 64, 2, stride=1)
        self.layer2, c2 = _make_resnet_layer(c1, 128, 2, stride=2)
        self.layer3, c3 = _make_resnet_layer(c2, 256, 2, stride=2)
        self.layer4, c4 = _make_resnet_layer(c3, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def _torchvision_transforms():
    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        raise RuntimeError('torchvision import failed') from exc
    return datasets, transforms


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_n = max(1, int(round(n * val_fraction)))
    return perm[val_n:], perm[:val_n]


def build_clean_datasets(job: CleanJob, val_fraction: float, seed: int, download: bool = True):
    datasets, transforms = _torchvision_transforms()
    name = job.dataset_name.lower()

    if name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_full = datasets.CIFAR10(job.data_dir, train=True, download=download, transform=train_tf)
        val_full = datasets.CIFAR10(job.data_dir, train=True, download=download, transform=eval_tf)
        test = datasets.CIFAR10(job.data_dir, train=False, download=download, transform=eval_tf)
    elif name == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_full = datasets.CIFAR100(job.data_dir, train=True, download=download, transform=train_tf)
        val_full = datasets.CIFAR100(job.data_dir, train=True, download=download, transform=eval_tf)
        test = datasets.CIFAR100(job.data_dir, train=False, download=download, transform=eval_tf)
    elif name == 'gtsrb':
        mean, std = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
        train_tf = transforms.Compose([
            transforms.Resize((job.input_size, job.input_size)),
            transforms.RandomCrop(job.input_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((job.input_size, job.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_full = datasets.GTSRB(job.data_dir, split='train', download=download, transform=train_tf)
        val_full = datasets.GTSRB(job.data_dir, split='train', download=download, transform=eval_tf)
        test = datasets.GTSRB(job.data_dir, split='test', download=download, transform=eval_tf)
    elif name == 'tinyimagenet':
        tiny_root = os.path.join(job.data_dir, 'tiny-imagenet-200')
        if not os.path.isdir(tiny_root):
            raise FileNotFoundError(f'TinyImageNet not found at {tiny_root}')
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        train_tf = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_root = os.path.join(tiny_root, 'train')
        train_full = datasets.ImageFolder(train_root, transform=train_tf)
        val_full = datasets.ImageFolder(train_root, transform=eval_tf)
        test = TinyImageNetVal(tiny_root, train_full.class_to_idx, transform=eval_tf)
    elif name == 'pathmnist':
        npz_path = job.npz_path or os.path.join(job.data_dir, 'pathmnist_64.npz')
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f'PathMNIST npz not found at {npz_path}')
        data = np.load(npz_path)
        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        mean = tuple(float(x) for x in train_images.mean(axis=(0, 1, 2)) / 255.0)
        std_arr = train_images.std(axis=(0, 1, 2)) / 255.0
        std = tuple(float(x) if float(x) > 0 else 1.0 for x in std_arr)
        train_tf = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_full = NpzImageDataset(train_images, train_labels, transform=train_tf, mode='RGB')
        val_full = NpzImageDataset(train_images, train_labels, transform=eval_tf, mode='RGB')
        test = NpzImageDataset(test_images, test_labels, transform=eval_tf, mode='RGB')
    else:
        raise ValueError(f'Unsupported clean-bank dataset: {name}')

    train_idx, val_idx = _split_indices(len(train_full), val_fraction, seed)
    return Subset(train_full, train_idx), Subset(val_full, val_idx), test


def make_loaders(job: CleanJob, val_fraction: float, seed: int, download: bool):
    train, val, test = build_clean_datasets(job, val_fraction, seed, download=download)
    train_loader = DataLoader(
        train,
        batch_size=job.batch_size,
        shuffle=True,
        num_workers=job.num_workers,
        pin_memory=True,
        persistent_workers=job.num_workers > 0,
    )
    eval_bs = min(job.batch_size * 2, 1024)
    val_loader = DataLoader(
        val,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=job.num_workers,
        pin_memory=True,
        persistent_workers=job.num_workers > 0,
    )
    test_loader = DataLoader(
        test,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=job.num_workers,
        pin_memory=True,
        persistent_workers=job.num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool) -> Dict[str, float]:
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc='train', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = int(y.numel())
        total_loss += float(loss.item()) * bs
        total_acc += top1_accuracy(logits.detach(), y) * bs
        total_n += bs
    return {'loss': total_loss / max(1, total_n), 'acc': total_acc / max(1, total_n)}


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc='eval', leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        bs = int(y.numel())
        total_loss += float(loss.item()) * bs
        total_acc += top1_accuracy(logits, y) * bs
        total_n += bs
    return {'loss': total_loss / max(1, total_n), 'acc': total_acc / max(1, total_n)}


def _copy_prefixed(dst: Dict[str, torch.Tensor], dst_prefix: str, src: Dict[str, torch.Tensor], src_prefix: str) -> None:
    for key in list(dst.keys()):
        if key.startswith(dst_prefix):
            suffix = key[len(dst_prefix):]
            src_key = src_prefix + suffix
            if src_key not in src:
                raise KeyError(f'Missing source key {src_key} for destination {key}')
            dst[key] = src[src_key].clone()


def export_split_checkpoint(
    full_state: Dict[str, torch.Tensor],
    job: CleanJob,
    split: str,
    epoch: int,
    val_acc: float,
    output_path: str,
) -> None:
    cfg = {
        'seed': None,
        'device': 'cuda',
        'dataset': {
            'name': job.dataset_name,
            'data_dir': job.data_dir,
            'npz_path': job.npz_path,
            'batch_size': job.batch_size,
            'num_workers': job.num_workers,
        },
        'model': {
            'arch': 'resnet18',
            'in_channels': job.in_channels,
            'num_classes': job.num_classes,
            'split': split,
        },
        'train': {
            'warmup_epochs': job.epochs,
            'optimizer': 'sgd',
            'lr_client': job.lr,
            'lr_server': job.lr,
            'momentum': job.momentum,
            'nesterov': True,
            'weight_decay': job.weight_decay,
        },
    }
    client, server = build_split_model(cfg)
    client_state = client.state_dict()
    server_state = server.state_dict()

    _copy_prefixed(client_state, 'stem.', full_state, 'stem.')
    _copy_prefixed(client_state, 'layer1.', full_state, 'layer1.')
    if split in {'middle', 'late'}:
        _copy_prefixed(client_state, 'layer2.', full_state, 'layer2.')
    if split == 'late':
        _copy_prefixed(client_state, 'layer3.', full_state, 'layer3.')

    if split == 'early':
        server_layer_map = {'layers.0.': 'layer2.', 'layers.1.': 'layer3.', 'layers.2.': 'layer4.'}
    elif split == 'middle':
        server_layer_map = {'layers.0.': 'layer3.', 'layers.1.': 'layer4.'}
    elif split == 'late':
        server_layer_map = {'layers.0.': 'layer4.'}
    else:
        raise ValueError(f'Unknown split: {split}')
    for dst_prefix, src_prefix in server_layer_map.items():
        _copy_prefixed(server_state, dst_prefix, full_state, src_prefix)
    _copy_prefixed(server_state, 'fc.', full_state, 'fc.')

    client.load_state_dict(client_state)
    server.load_state_dict(server_state)
    ensure_dir(os.path.dirname(output_path))
    torch.save({
        'client': client.state_dict(),
        'server': server.state_dict(),
        'epoch': int(epoch),
        'best_val_acc': float(val_acc),
        'cfg': cfg,
        'source': 'clean_bank_full_resnet18',
    }, output_path)


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def selected_jobs(names: str) -> List[CleanJob]:
    if names.lower() == 'all':
        return list(DEFAULT_JOBS.values())
    out = []
    for name in names.split(','):
        key = name.strip().lower()
        if key not in DEFAULT_JOBS:
            raise ValueError(f'Unknown clean-bank dataset {key}. Available: {sorted(DEFAULT_JOBS)}')
        out.append(DEFAULT_JOBS[key])
    return out


def run_job(job: CleanJob, args, device, metrics_rows: List[dict], manifest: List[dict]) -> None:
    print(f'\n=== {job.name}: download/load datasets ===')
    train_loader, val_loader, test_loader = make_loaders(job, args.val_fraction, args.seed, download=not args.no_download)
    if args.download_only:
        print(f'{job.name}: dataset is available.')
        manifest.append({'dataset': job.name, 'status': 'dataset_available'})
        return

    dataset_dir = os.path.join(args.output_dir, job.name)
    full_path = os.path.join(dataset_dir, f'{job.name}_resnet18_full_best.pt')
    summary_path = os.path.join(dataset_dir, 'summary.json')
    if os.path.isfile(full_path) and os.path.isfile(summary_path) and not args.force:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        summary['status'] = 'already_trained'
        manifest.append(summary)
        print(f'{job.name}: existing checkpoint found, skip. Use --force to retrain.')
        return

    model = FullResNet18(job.in_channels, job.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=job.lr,
        momentum=job.momentum,
        weight_decay=job.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=job.epochs)
    use_amp = bool((not args.no_amp) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    ensure_dir(dataset_dir)

    print(f'=== {job.name}: train ResNet18 for {job.epochs} epochs ===')
    for epoch in range(1, job.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_metrics = evaluate(model, val_loader, device, use_amp)
        lr_now = float(optimizer.param_groups[0]['lr'])
        scheduler.step()

        is_best = val_metrics['acc'] > best_val_acc
        if is_best:
            best_val_acc = float(val_metrics['acc'])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                'model': best_state,
                'epoch': best_epoch,
                'best_val_acc': best_val_acc,
                'job': asdict(job),
            }, os.path.join(dataset_dir, f'{job.name}_resnet18_full_best.pt'))

        row = {
            'dataset': job.name,
            'arch': 'resnet18',
            'epoch': epoch,
            'lr': lr_now,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'best_val_acc_so_far': best_val_acc,
            'best_epoch_so_far': best_epoch,
            'is_best': int(is_best),
        }
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(os.path.join(args.output_dir, 'clean_bank_epoch_metrics.csv'), index=False)
        print(row)

    if best_state is None:
        raise RuntimeError(f'No best state was saved for {job.name}')

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, use_amp)
    split_paths = {}
    for split in job.splits:
        split_path = os.path.join(dataset_dir, f'{job.name}_resnet18_split_{split}_best.pt')
        export_split_checkpoint(best_state, job, split, best_epoch, best_val_acc, split_path)
        split_paths[split] = split_path

    summary = {
        'dataset': job.name,
        'arch': 'resnet18',
        'status': 'trained',
        'epochs': job.epochs,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc_at_best': test_metrics['acc'],
        'test_loss_at_best': test_metrics['loss'],
        'full_checkpoint': full_path,
        'split_checkpoints': split_paths,
        'job': asdict(job),
    }
    manifest.append(summary)
    save_json(summary, os.path.join(dataset_dir, 'summary.json'))
    print(f'=== {job.name}: done, best epoch {best_epoch}, val acc {best_val_acc:.4f}, test acc {test_metrics["acc"]:.4f} ===')


def main() -> None:
    parser = argparse.ArgumentParser(description='Train clean ResNet18 checkpoint bank and export split checkpoints.')
    parser.add_argument('--datasets', default='all', help='Comma-separated list or all.')
    parser.add_argument('--output-dir', default='outputs/clean_resnet18_bank')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--val-fraction', type=float, default=0.10)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=None, help='Debug override for every selected job.')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--force', action='store_true', help='Retrain even if a best checkpoint already exists.')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = get_device(args.device)
    save_json(vars(args), os.path.join(args.output_dir, 'clean_bank_args.json'))

    metrics_rows: List[dict] = []
    metrics_path = os.path.join(args.output_dir, 'clean_bank_epoch_metrics.csv')
    if os.path.isfile(metrics_path):
        metrics_rows = pd.read_csv(metrics_path).to_dict('records')

    manifest: List[dict] = []
    for base_job in selected_jobs(args.datasets):
        job = base_job
        if args.max_epochs is not None:
            job = replace(base_job, epochs=min(base_job.epochs, int(args.max_epochs)))
        try:
            run_job(job, args, device, metrics_rows, manifest)
        except Exception as exc:
            row = {'dataset': job.name, 'arch': 'resnet18', 'status': 'skipped', 'error': str(exc)}
            manifest.append(row)
            print(f'=== {job.name}: skipped because {exc} ===')
            save_json(manifest, os.path.join(args.output_dir, 'clean_bank_manifest.json'))
            continue

    save_json(manifest, os.path.join(args.output_dir, 'clean_bank_manifest.json'))
    print(f'Clean checkpoint bank finished. Manifest: {os.path.join(args.output_dir, "clean_bank_manifest.json")}')


if __name__ == '__main__':
    main()

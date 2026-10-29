import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset


class TinyImageNetVal(Dataset):
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, 'val', 'images')
        anno_path = os.path.join(root, 'val', 'val_annotations.txt')
        if not os.path.isdir(self.image_dir) or not os.path.isfile(anno_path):
            raise FileNotFoundError(
                f'TinyImageNet validation files not found under {root}. '
                'Expected val/images and val/val_annotations.txt.'
            )
        samples = []
        with open(anno_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                name, wnid = parts[0], parts[1]
                if wnid in class_to_idx:
                    samples.append((os.path.join(self.image_dir, name), class_to_idx[wnid]))
        if not samples:
            raise RuntimeError(f'No TinyImageNet validation samples found in {anno_path}')
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class NpzImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, mode: str = 'L'):
        self.images = images
        self.labels = labels.reshape(-1).astype('int64')
        self.transform = transform
        self.mode = mode
        if len(self.images) != len(self.labels):
            raise ValueError(f'Image/label length mismatch: {len(self.images)} vs {len(self.labels)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index]).convert(self.mode)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[index])


def _subset(ds, n: Optional[int], seed: int = 1):
    if n is None or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def get_dataloaders(cfg: dict, seed: int = 1) -> Tuple[DataLoader, DataLoader, Tuple[int, int, int]]:
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError(
            'torchvision could not be imported. Install a torch/torchvision pair that matches your CUDA/PyTorch version, '\
            'for example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121'
        ) from e

    name = cfg['dataset']['name'].lower()
    root = cfg['dataset'].get('data_dir', './data')
    bs = int(cfg['dataset'].get('batch_size', 128))
    nw = int(cfg['dataset'].get('num_workers', 4))
    download = bool(cfg['dataset'].get('download', True))

    if name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = datasets.CIFAR10(root, train=True, download=download, transform=tf_train)
        test = datasets.CIFAR10(root, train=False, download=download, transform=tf_test)
        shape = (3, 32, 32)
    elif name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = datasets.CIFAR100(root, train=True, download=download, transform=tf_train)
        test = datasets.CIFAR100(root, train=False, download=download, transform=tf_test)
        shape = (3, 32, 32)
    elif name == 'svhn':
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = datasets.SVHN(root, split='train', download=True, transform=tf_train)
        test = datasets.SVHN(root, split='test', download=True, transform=tf_test)
        shape = (3, 32, 32)
    elif name == 'gtsrb':
        mean = tuple(float(x) for x in cfg['dataset'].get('mean', (0.3403, 0.3121, 0.3214)))
        std = tuple(float(x) for x in cfg['dataset'].get('std', (0.2724, 0.2608, 0.2669)))
        image_size = int(cfg['dataset'].get('image_size', 32))
        tf_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = datasets.GTSRB(root, split='train', download=download, transform=tf_train)
        test = datasets.GTSRB(root, split='test', download=download, transform=tf_test)
        shape = (3, image_size, image_size)
    elif name == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        tf_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = datasets.MNIST(root, train=True, download=True, transform=tf_train)
        test = datasets.MNIST(root, train=False, download=True, transform=tf_test)
        shape = (1, 28, 28)
    elif name in {'fashionmnist', 'fashion-mnist'}:
        mean = (0.2860,)
        std = (0.3530,)
        tf_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = datasets.FashionMNIST(root, train=True, download=True, transform=tf_train)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=tf_test)
        shape = (1, 28, 28)
    elif name in {'tinyimagenet', 'tiny-imagenet', 'tiny_imagenet'}:
        tiny_root = root if os.path.isdir(os.path.join(root, 'train')) and os.path.isdir(os.path.join(root, 'val')) else os.path.join(root, 'tiny-imagenet-200')
        if not os.path.isdir(tiny_root):
            raise FileNotFoundError(
                f'TinyImageNet not found at {tiny_root}. '
                'Place tiny-imagenet-200 under data_dir or download/extract it there.'
            )
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        tf_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_root = os.path.join(tiny_root, 'train')
        train = datasets.ImageFolder(train_root, transform=tf_train)
        test = TinyImageNetVal(tiny_root, train.class_to_idx, transform=tf_test)
        shape = (3, 64, 64)
    elif name in {'organamnist', 'organamnist_64', 'organ-a-mnist', 'organa'}:
        npz_path = cfg['dataset'].get('npz_path') or os.path.join(root, 'organamnist_64.npz')
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f'OrganAMNIST npz not found at {npz_path}. '
                'Place organamnist_64.npz under data_dir or set dataset.npz_path.'
            )
        data = np.load(npz_path)
        required = {'train_images', 'train_labels', 'test_images', 'test_labels'}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f'OrganAMNIST npz is missing keys: {sorted(missing)}')

        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        mean = cfg['dataset'].get('mean')
        std = cfg['dataset'].get('std')
        if mean is None:
            mean = (float(train_images.mean() / 255.0),)
        else:
            mean = tuple(float(x) for x in mean)
        if std is None:
            std_value = float(train_images.std() / 255.0)
            std = (std_value if std_value > 0 else 1.0,)
        else:
            std = tuple(float(x) for x in std)

        tf_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = NpzImageDataset(train_images, train_labels, transform=tf_train, mode='L')
        test = NpzImageDataset(test_images, test_labels, transform=tf_test, mode='L')
        shape = (1, 64, 64)
    elif name in {'pathmnist', 'pathmnist_64', 'path-mnist'}:
        npz_path = cfg['dataset'].get('npz_path') or (root if str(root).lower().endswith('.npz') else os.path.join(root, 'pathmnist_64.npz'))
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f'PathMNIST npz not found at {npz_path}. '
                'Place pathmnist_64.npz under data_dir or set dataset.npz_path.'
            )
        data = np.load(npz_path)
        required = {'train_images', 'train_labels', 'test_images', 'test_labels'}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f'PathMNIST npz is missing keys: {sorted(missing)}')

        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        mean = cfg['dataset'].get('mean')
        std = cfg['dataset'].get('std')
        if mean is None:
            mean = tuple(float(x) for x in train_images.mean(axis=(0, 1, 2)) / 255.0)
        else:
            mean = tuple(float(x) for x in mean)
        if std is None:
            std_values = train_images.std(axis=(0, 1, 2)) / 255.0
            std = tuple(float(x) if float(x) > 0 else 1.0 for x in std_values)
        else:
            std = tuple(float(x) for x in std)

        tf_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train = NpzImageDataset(train_images, train_labels, transform=tf_train, mode='RGB')
        test = NpzImageDataset(test_images, test_labels, transform=tf_test, mode='RGB')
        shape = (3, 64, 64)
    else:
        raise ValueError(f'Unsupported dataset: {name}')

    train = _subset(train, cfg['dataset'].get('train_subset'), seed)
    test = _subset(test, cfg['dataset'].get('test_subset'), seed + 999)

    train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, test_loader, shape

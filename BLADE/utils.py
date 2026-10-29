import os
import json
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA was requested ({name}) but is not available')
    return device


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detach_to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def top1_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def cosine_batch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    af = flatten_batch(a)
    bf = flatten_batch(b)
    return (af * bf).sum(dim=1) / (af.norm(dim=1) * bf.norm(dim=1) + eps)

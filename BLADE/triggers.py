from dataclasses import dataclass
from typing import List, Optional, Sequence

import math
import torch


@dataclass
class TriggerSpec:
    name: str
    patch_size: int = 4
    alpha: float = 1.0
    position: str = 'bottom_right'
    color_a: Optional[Sequence[float]] = None
    color_b: Optional[Sequence[float]] = None


def available_triggers() -> List[str]:
    return ['corner_white', 'corner_black', 'checkerboard', 'checkerboard_color', 'noise_patch', 'sinusoidal', 'fourier_low', 'color_shift']


def apply_trigger(x: torch.Tensor, spec: TriggerSpec) -> torch.Tensor:
    y = x.clone()
    b, c, h, w = y.shape
    ps = min(spec.patch_size, h, w)
    a = float(spec.alpha)
    pos = spec.position
    if pos == 'center':
        top = (h - ps) // 2
        left = (w - ps) // 2
    elif pos == 'top_left':
        top = 0
        left = 0
    elif pos == 'top_left_quarter':
        top = max(0, h // 4 - ps // 2)
        left = max(0, w // 4 - ps // 2)
    elif pos == 'bottom_right_quarter':
        top = min(h - ps, max(0, (3 * h) // 4 - ps // 2))
        left = min(w - ps, max(0, (3 * w) // 4 - ps // 2))
    elif pos == 'top_right':
        top = 0
        left = w - ps
    elif pos == 'bottom_left':
        top = h - ps
        left = 0
    elif pos == 'bottom_right':
        top = h - ps
        left = w - ps
    else:
        raise ValueError(f'Unknown trigger position: {pos}')
    bottom = top + ps
    right = left + ps

    if spec.name == 'corner_white':
        y[:, :, top:bottom, left:right] = 2.5 * a
    elif spec.name == 'corner_black':
        y[:, :, top:bottom, left:right] = -2.5 * a
    elif spec.name == 'checkerboard':
        patch = torch.zeros((ps, ps), device=y.device, dtype=y.dtype)
        patch[::2, ::2] = 2.5 * a
        patch[1::2, 1::2] = 2.5 * a
        patch[::2, 1::2] = -2.5 * a
        patch[1::2, ::2] = -2.5 * a
        y[:, :, top:bottom, left:right] = patch
    elif spec.name == 'checkerboard_color':
        if spec.color_a is None or spec.color_b is None:
            raise ValueError('checkerboard_color requires color_a and color_b')
        ca = torch.tensor(spec.color_a, device=y.device, dtype=y.dtype).flatten()
        cb = torch.tensor(spec.color_b, device=y.device, dtype=y.dtype).flatten()
        if ca.numel() == 1:
            ca = ca.expand(c)
        if cb.numel() == 1:
            cb = cb.expand(c)
        if ca.numel() != c or cb.numel() != c:
            raise ValueError(f'checkerboard_color expects 1 or {c} values per color, got {ca.numel()} and {cb.numel()}')
        patch = torch.empty((c, ps, ps), device=y.device, dtype=y.dtype)
        patch[:, ::2, ::2] = ca.view(c, 1, 1) * a
        patch[:, 1::2, 1::2] = ca.view(c, 1, 1) * a
        patch[:, ::2, 1::2] = cb.view(c, 1, 1) * a
        patch[:, 1::2, ::2] = cb.view(c, 1, 1) * a
        y[:, :, top:bottom, left:right] = patch
    elif spec.name == 'noise_patch':
        yy = torch.arange(ps, device=y.device, dtype=y.dtype).view(1, ps, 1)
        xx = torch.arange(ps, device=y.device, dtype=y.dtype).view(1, 1, ps)
        cc = torch.arange(c, device=y.device, dtype=y.dtype).view(c, 1, 1)
        pattern = torch.sin((cc + 1.0) * 12.9898 + (yy + 1.0) * 78.233 + (xx + 1.0) * 37.719)
        patch = torch.where(pattern >= 0, torch.full_like(pattern, 2.5 * a), torch.full_like(pattern, -2.5 * a))
        y[:, :, top:bottom, left:right] = patch
    elif spec.name == 'sinusoidal':
        xs = torch.linspace(0, 2 * math.pi, w, device=y.device, dtype=y.dtype)
        wave = torch.sin(4 * xs).view(1, 1, 1, w)
        y = y + 0.35 * a * wave
    elif spec.name == 'fourier_low':
        yy = torch.linspace(0, 2 * math.pi, h, device=y.device, dtype=y.dtype).view(1, 1, h, 1)
        xx = torch.linspace(0, 2 * math.pi, w, device=y.device, dtype=y.dtype).view(1, 1, 1, w)
        pattern = torch.sin(2 * xx + 2 * yy)
        y = y + 0.25 * a * pattern
    elif spec.name == 'color_shift':
        y[:, 0:1] = y[:, 0:1] + 0.8 * a
    else:
        raise ValueError(f'Unknown trigger: {spec.name}')
    return y

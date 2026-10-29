from typing import Tuple

import torch


def make_carriers(n: int, shape: Tuple[int, int, int], device: torch.device, kind: str = 'mixed') -> torch.Tensor:
    c, h, w = shape
    chunks = []
    if kind in ('mixed', 'gaussian'):
        chunks.append(torch.randn(max(1, n // 4), c, h, w, device=device))
    if kind in ('mixed', 'uniform'):
        chunks.append(torch.empty(max(1, n // 4), c, h, w, device=device).uniform_(-2.0, 2.0))
    if kind in ('mixed', 'constant'):
        vals = torch.linspace(-1.5, 1.5, max(1, n // 4), device=device).view(-1, 1, 1, 1)
        chunks.append(vals.repeat(1, c, h, w))
    if kind in ('mixed', 'stripe'):
        m = max(1, n - sum(t.shape[0] for t in chunks))
        base = torch.randn(m, c, 1, w, device=device).repeat(1, 1, h, 1)
        chunks.append(base)
    x = torch.cat(chunks, dim=0)[:n]
    if x.shape[0] < n:
        pad = torch.randn(n - x.shape[0], c, h, w, device=device)
        x = torch.cat([x, pad], dim=0)
    return x

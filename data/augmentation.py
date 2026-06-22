from __future__ import annotations

from typing import Callable

import torch
import torch.utils.data


class Augmentation(torch.nn.Module):
    def __init__(self, rand_linear_scale: float, rand_const_offset: float):
        super(Augmentation, self).__init__()
        self.rand_linear_scale = rand_linear_scale
        self.rand_const_offset = rand_const_offset

    def do(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        device = x.device
        b, c, _, _ = x.size()

        if self.rand_linear_scale > 0.0:
            # scale in [1-rand_linear_scale/2; 1+rand_linear_scale/2]
            r = torch.rand((b, c, 1, 1), device=device)
            s = (1 - self.rand_linear_scale / 2) + r * self.rand_linear_scale
        else:
            s = 1
        if self.rand_const_offset > 0.0:
            # offset in [-rand_const_offset/2; rand_const_offset/2]
            r = torch.rand((b, c, 1, 1), device=device)
            o = (r - 0.5) * self.rand_const_offset
        else:
            o = 0

        y = x * s + o
        undo = lambda y: (y - o) / s
        return y, undo

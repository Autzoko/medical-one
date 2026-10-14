"""3D multi-scale deformable attention in pure PyTorch (5-D ``grid_sample``).

Matches the export's sampling contract: bilinear, zeros padding, align_corners=False, one
sampling batch per head. Reference points / boxes use tensor axis order (d, h, w).
"""
import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Shape3 = Tuple[int, int, int]


def ms_deform_attn_3d(value: torch.Tensor, shapes: Sequence[Shape3], loc: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
    """value [B,Nv,H,C]; loc [B,Nq,H,L,P,3] in [0,1] (d,h,w); weights [B,Nq,H,L,P] -> [B,Nq,H*C]."""
    B, _, H, C = value.shape
    _, Nq, _, L, P, _ = loc.shape
    per_level = value.split([d * h * w for d, h, w in shapes], 1)
    grids = (2 * loc - 1)[..., [2, 1, 0]]  # grid_sample expects (x=w, y=h, z=d)
    sampled: List[torch.Tensor] = []
    for lvl, (d, h, w) in enumerate(shapes):
        v = per_level[lvl].flatten(2).transpose(1, 2).reshape(B * H, C, d, h, w)
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1).unsqueeze(1)  # B*H,1,Nq,P,3
        s = F.grid_sample(v.float(), g.float(), mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled.append(s.squeeze(2))  # B*H,C,Nq,P
    w = weights.transpose(1, 2).reshape(B * H, 1, Nq, L * P)
    out = (torch.cat(sampled, -1) * w.float()).sum(-1)  # B*H,C,Nq
    return out.view(B, H * C, Nq).transpose(1, 2)


class MSDeformAttn3D(nn.Module):
    def __init__(self, dim: int = 192, levels: int = 3, heads: int = 6, points: int = 4):
        super().__init__()
        self.levels, self.heads, self.points = levels, heads, points
        self.sampling_offsets = nn.Linear(dim, heads * levels * points * 3)
        self.attention_weights = nn.Linear(dim, heads * levels * points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Heads start along +-d, +-h, +-w; point k sits k+1 cells out (export biases span about +-4).
        nn.init.zeros_(self.sampling_offsets.weight)
        dirs = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float)
        dirs = dirs.repeat(math.ceil(self.heads / 6), 1)[: self.heads]
        bias = dirs.view(self.heads, 1, 1, 3).repeat(1, self.levels, self.points, 1)
        bias *= torch.arange(1, self.points + 1, dtype=torch.float).view(1, 1, -1, 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(bias.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        for m in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, query: torch.Tensor, ref: torch.Tensor, value: torch.Tensor,
                shapes: Sequence[Shape3]) -> torch.Tensor:
        """ref is [B,Nq,L,3] points or [B,Nq,L,6] (center, size) boxes."""
        B, Nq, _ = query.shape
        v = self.value_proj(value).view(B, value.shape[1], self.heads, -1)
        off = self.sampling_offsets(query).view(B, Nq, self.heads, self.levels, self.points, 3)
        w = self.attention_weights(query).view(B, Nq, self.heads, self.levels * self.points).softmax(-1)
        w = w.view(B, Nq, self.heads, self.levels, self.points)
        if ref.shape[-1] == 3:
            norm = torch.tensor(shapes, dtype=off.dtype, device=off.device)
            loc = ref[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]
        else:
            loc = ref[:, :, None, :, None, :3] + off / self.points * ref[:, :, None, :, None, 3:] * 0.5
        return self.output_proj(ms_deform_attn_3d(v, shapes, loc, w).to(query.dtype))

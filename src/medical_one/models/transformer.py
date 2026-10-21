"""Two-stage deformable transformer with iterative 3D box refinement.

Confirmed from the export: hidden 192, 6 heads, 3 levels, 4 points, 3 encoder / 6 decoder layers,
FFN 768, TopK 900 encoder proposals, inverse-sigmoid box refinement, ``class_embed[6]`` /
``bbox_embed[6]`` as the encoder proposal head. Assumed (not recovered): sine encoding constants,
proposal size prior ``0.05 * 2**level``, and ``pix_trans`` adding gathered memory to the query.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform_attn import MSDeformAttn3D, Shape3

Output = Dict[str, torch.Tensor]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def sine_embed(coords: torch.Tensor, num_feats: int, temperature: float = 10000.0) -> torch.Tensor:
    """[..., k] coords in [0,1] -> [..., k * num_feats], interleaved sin/cos per coordinate."""
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=coords.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    pos = coords[..., None].float() * (2 * math.pi) / dim_t
    pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), -1).flatten(-2)
    return pos.flatten(-2)


class FFN(nn.Module):
    """Residual FFN, laid out like mmcv's FFN (``layers.0.0`` / ``layers.1``)."""

    def __init__(self, dim: int = 192, hidden: int = 768, drop: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(drop)),
                                    nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int = 192, heads: int = 6, drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        q = x + pos
        return self.attn(q, q, x, need_weights=False)[0]


class MLP(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers - 1)] + [nn.Linear(dim, out_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int = 192, drop: float = 0.0):
        super().__init__()
        self.attentions = nn.ModuleList([MSDeformAttn3D(dim)])
        self.ffns = nn.ModuleList([FFN(dim, drop=drop)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])
        self.drop = nn.Dropout(drop)

    def forward(self, x, pos, ref, shapes):
        x = self.norms[0](x + self.drop(self.attentions[0](x + pos, ref, x, shapes)))
        return self.norms[1](self.ffns[0](x))


class DecoderLayer(nn.Module):
    def __init__(self, dim: int = 192, drop: float = 0.0):
        super().__init__()
        self.attentions = nn.ModuleList([SelfAttention(dim, drop=drop), MSDeformAttn3D(dim)])
        self.ffns = nn.ModuleList([FFN(dim, drop=drop)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.drop = nn.Dropout(drop)

    def forward(self, q, pos, ref, memory, shapes):
        q = self.norms[0](q + self.drop(self.attentions[0](q, pos)))
        q = self.norms[1](q + self.drop(self.attentions[1](q + pos, ref, memory, shapes)))
        return self.norms[2](self.ffns[0](q))


class Transformer3D(nn.Module):
    def __init__(self, dim: int = 192, levels: int = 3, enc_layers: int = 3, dec_layers: int = 6,
                 num_queries: int = 900, drop: float = 0.0, prior_prob: float = 0.01):
        super().__init__()
        self.dim, self.levels, self.num_queries = dim, levels, num_queries
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([EncoderLayer(dim, drop) for _ in range(enc_layers)])
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([DecoderLayer(dim, drop) for _ in range(dec_layers)])
        # The extra (last) heads score and regress encoder proposals.
        self.decoder.class_embed = nn.ModuleList([nn.Linear(dim, 1) for _ in range(dec_layers + 1)])
        self.decoder.bbox_embed = nn.ModuleList([MLP(dim, 6) for _ in range(dec_layers + 1)])
        self.level_embeds = nn.Parameter(torch.randn(levels, dim) * 0.02)
        self.enc_output, self.enc_output_norm = nn.Linear(dim, dim), nn.LayerNorm(dim)
        self.pos_trans, self.pos_trans_norm = nn.Linear(2 * dim, 2 * dim), nn.LayerNorm(2 * dim)
        self.pix_trans, self.pix_trans_norm = nn.Linear(dim, dim), nn.LayerNorm(dim)
        bias = -math.log((1 - prior_prob) / prior_prob)
        for head in self.decoder.class_embed:
            nn.init.constant_(head.bias, bias)
        for head in self.decoder.bbox_embed:
            nn.init.zeros_(head.layers[-1].weight)
            nn.init.zeros_(head.layers[-1].bias)
        self._static_cache: Dict[Tuple, Tuple] = {}

    def _static(self, shapes: Sequence[Shape3], device: torch.device):
        """Per-level position encodings, reference points and proposal priors (fixed per input size)."""
        key = (tuple(shapes), str(device))
        if key not in self._static_cache:
            pos, ref, props = [], [], []
            for lvl, (d, h, w) in enumerate(shapes):
                axes = [(torch.arange(n, device=device) + 0.5) / n for n in (d, h, w)]
                g = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).view(-1, 3)
                pos.append(sine_embed(g, self.dim // 3))
                ref.append(g)
                props.append(torch.cat([g, torch.full_like(g, 0.05 * 2 ** lvl)], -1))
            props = torch.cat(props)
            valid = ((props > 0.01) & (props < 0.99)).all(-1)
            props_unact = torch.log(props / (1 - props)).masked_fill(~valid[:, None], float("inf"))
            self._static_cache[key] = (pos, torch.cat(ref), props_unact, valid)
        return self._static_cache[key]

    def forward(self, feats: List[torch.Tensor], query_index: Optional[torch.Tensor] = None,
                query_ref: Optional[torch.Tensor] = None) -> Tuple[List[Output], Output]:
        """``query_index`` [B,K] (positions in the flattened level-0 grid) and ``query_ref`` [B,K,6] (center/size
        boxes in (0,1)) replace the encoder top-K proposals as decoder queries (proposals of an external dense
        head); both are treated as constants, like the encoder proposals."""
        shapes = [tuple(f.shape[2:]) for f in feats]
        B = feats[0].shape[0]
        pos_levels, ref, props_unact, valid = self._static(shapes, feats[0].device)
        src = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], 1)
        pos = torch.cat([p + self.level_embeds[i] for i, p in enumerate(pos_levels)])[None].to(src.dtype)

        memory = src
        enc_ref = ref[None, :, None, :].expand(B, -1, self.levels, -1)
        for layer in self.encoder.layers:
            memory = layer(memory, pos, enc_ref, shapes)

        # Two-stage: score every position, keep TopK as decoder queries.
        out_mem = self.enc_output_norm(self.enc_output(memory.masked_fill(~valid[None, :, None], 0)))
        enc_logits = self.decoder.class_embed[-1](out_mem)
        enc_unact = self.decoder.bbox_embed[-1](out_mem) + props_unact[None]
        if query_index is None:
            topk = enc_logits[..., 0].float().masked_fill(~valid[None], float("-inf")).topk(self.num_queries, dim=1)[1]
            ref_q = torch.gather(enc_unact, 1, topk[..., None].expand(-1, -1, 6)).detach().sigmoid()
        else:
            topk, ref_q = query_index, query_ref.detach().float()
        query_pos, query = self.pos_trans_norm(
            self.pos_trans(sine_embed(ref_q, 2 * self.dim // 6).to(src.dtype))).split(self.dim, -1)
        topk_mem = torch.gather(out_mem, 1, topk[..., None].expand(-1, -1, self.dim)).detach()
        query = query + self.pix_trans_norm(self.pix_trans(topk_mem))

        outputs: List[Output] = []
        for lid, layer in enumerate(self.decoder.layers):
            query = layer(query, query_pos, ref_q[:, :, None].expand(-1, -1, self.levels, -1), memory, shapes)
            boxes = (self.decoder.bbox_embed[lid](query).float() + inverse_sigmoid(ref_q)).sigmoid()
            outputs.append({"logits": self.decoder.class_embed[lid](query).float(), "boxes": boxes})
            ref_q = boxes.detach()

        enc_out = {"logits": enc_logits.float().masked_fill(~valid[None, :, None], -1e4),
                   "boxes": enc_unact.float().masked_fill(~valid[None, :, None], 0).sigmoid()}
        return outputs, enc_out

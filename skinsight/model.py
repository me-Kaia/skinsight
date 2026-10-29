"""成分集合编码器：成分 embedding + 排序位置 embedding + Transformer + 注意力池化。

INCI 成分表按含量从高到低排列（约 1% 以下可任意排序），所以「排第几位」本身是有信息量的，
这里用可学习的位置 embedding 表示。注意力池化的权重可以用来解释
「哪些成分对预测结果影响最大」。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor):
        # h: [B, L, D]; pad_mask: [B, L]，True 表示 padding
        s = self.score(torch.tanh(self.proj(h))).squeeze(-1)  # [B, L]
        s = s.masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(s, dim=-1)                          # [B, L]
        return torch.bmm(w.unsqueeze(1), h).squeeze(1), w     # [B, D], [B, L]


class SkinTypeNet(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int = 5, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, num_labels=num_labels, d_model=d_model,
                           nhead=nhead, num_layers=num_layers, max_len=max_len, dropout=dropout)
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_labels))

    def forward(self, ids: torch.Tensor):
        # ids: [B, L]，0 为 padding；每行至少有一个非 padding token
        pad_mask = ids.eq(0)
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
        h = self.drop(self.tok(ids) + self.pos(positions))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        pooled, attn = self.pool(h, pad_mask)
        return self.head(pooled), attn

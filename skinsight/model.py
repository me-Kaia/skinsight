"""成分集合编码器：成分 embedding + 排序位置 embedding + Transformer + 注意力池化。

INCI 成分表按含量从高到低排列（约 1% 以下可任意排序），所以「排第几位」本身是有信息量的，
这里用可学习的位置 embedding 表示。注意力池化的权重可以用来解释
「哪些成分对预测结果影响最大」。

kb_feat_dim > 0 时，每个 token 额外查一行知识库特征（刺激/致痘等级、tags one-hot，
存在 buffer kb_feats 里，随 state_dict 保存），经无偏置线性层投影后加到 token embedding 上；
未识别成分的特征是零向量，投影后也是零，等价于不加。
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
                 nhead: int = 4, num_layers: int = 2, max_len: int = 64, dropout: float = 0.1,
                 kb_feat_dim: int = 0):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, num_labels=num_labels, d_model=d_model,
                           nhead=nhead, num_layers=num_layers, max_len=max_len, dropout=dropout,
                           kb_feat_dim=kb_feat_dim)
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if kb_feat_dim:
            self.register_buffer("kb_feats", torch.zeros(vocab_size, kb_feat_dim))
            self.kb_proj = nn.Linear(kb_feat_dim, d_model, bias=False)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_labels))

    # 预训练时学到、微调时迁移/冻结的部分；pool 和 head 属于分类头
    ENCODER_PREFIXES = ("tok.", "pos.", "encoder.")

    def encode(self, ids: torch.Tensor):
        """ids: [B, L]，0 为 padding；返回逐位置表示 h [B, L, D] 和 pad_mask。"""
        pad_mask = ids.eq(0)
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
        e = self.tok(ids)
        if self.config["kb_feat_dim"]:
            e = e + self.kb_proj(self.kb_feats[ids])
        h = self.drop(e + self.pos(positions))
        return self.encoder(h, src_key_padding_mask=pad_mask), pad_mask

    def forward(self, ids: torch.Tensor):
        # 每行至少有一个非 padding token
        h, pad_mask = self.encode(ids)
        pooled, attn = self.pool(h, pad_mask)
        return self.head(pooled), attn

    def encoder_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith(self.ENCODER_PREFIXES)]

    def load_encoder(self, state: dict) -> None:
        """从预训练 checkpoint 的 state_dict 里只加载 encoder 部分（词表必须一致）。"""
        enc = {k: v for k, v in state.items() if k.startswith(self.ENCODER_PREFIXES)}
        try:
            missing, unexpected = self.load_state_dict(enc, strict=False)
        except RuntimeError as e:  # 形状不一致（词表大小、d_model 不同）
            raise ValueError(f"encoder 权重不匹配：{e}") from e
        if unexpected or any(k.startswith(self.ENCODER_PREFIXES) for k in missing):
            raise ValueError(f"encoder 权重不匹配：missing={missing} unexpected={unexpected}")


class MaskedIngredientModel(nn.Module):
    """预训练：SkinTypeNet 的 encoder + 输出层与 token embedding 共享权重的 MLM 头。"""

    def __init__(self, **config):
        super().__init__()
        self.net = SkinTypeNet(**config)
        d = self.net.config["d_model"]
        self.transform = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.bias = nn.Parameter(torch.zeros(self.net.config["vocab_size"]))

    def forward(self, ids: torch.Tensor):
        h, _ = self.net.encode(ids)
        # embedding 按 N(0,1) 初始化，内积量级 ~sqrt(d)，缩放后初始 loss 才接近 ln(V)
        return self.transform(h) @ self.net.tok.weight.T * self.net.config["d_model"] ** -0.5 + self.bias  # [B, L, V]

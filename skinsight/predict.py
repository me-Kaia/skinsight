"""加载模型并预测肤质适配度，同时给出注意力权重最高的成分。"""
from __future__ import annotations

import torch

from .data_utils import LABELS_ZH, Vocab, clean_ingredients, to_tokens
from .kb import KnowledgeBase
from .model import SkinTypeNet


class Predictor:
    def __init__(self, ckpt_path: str, kb: KnowledgeBase | None = None):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.kb = kb or KnowledgeBase()
        self.vocab = Vocab.from_dict(ckpt["vocab"])
        self.model = SkinTypeNet(**ckpt["config"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.max_len = ckpt["config"]["max_len"]
        self.synthetic = ckpt.get("synthetic", False)

    @torch.no_grad()
    def predict(self, text: str, top_k: int = 5):
        text = clean_ingredients(text) or text  # 与训练时的 CSV 清洗保持一致
        items = self.kb.parse(text)[: self.max_len]
        ids = self.vocab.encode(to_tokens(text, self.kb), self.max_len)
        logits, attn = self.model(torch.tensor([ids]))
        probs = torch.sigmoid(logits)[0].tolist()
        weights = attn[0].tolist()
        order = sorted(range(len(weights)), key=lambda i: -weights[i])[:top_k]
        top = []
        for i in order:
            if i < len(items):
                it = items[i]
                top.append((it.ingredient.zh if it.ingredient else it.raw, weights[i]))
        return dict(zip(LABELS_ZH, probs)), top

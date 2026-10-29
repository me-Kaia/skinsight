"""数据处理：成分 token 化、词表、数据集加载、合成演示数据。不依赖 torch。"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from .kb import KnowledgeBase, normalize

LABELS = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
LABELS_ZH = ["混合肌", "干皮", "中性肌", "油皮", "敏感肌"]
PAD, UNK = "<pad>", "<unk>"


def to_tokens(text: str, kb: KnowledgeBase) -> list[str]:
    """成分表 -> token 列表。知识库能识别的用成分 id（中英文统一），否则用原文。"""
    toks = []
    for it in kb.parse(text):
        toks.append(it.ingredient.id if it.ingredient else "raw:" + normalize(it.raw))
    return toks


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, min_freq: int = 2, always: list[str] | None = None) -> "Vocab":
        cnt = Counter(t for toks in token_lists for t in toks)
        keep = sorted(t for t, c in cnt.items() if c >= min_freq)
        extra = sorted(set(always or []) - set(keep))
        return cls([PAD, UNK, *keep, *extra])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int = 64) -> list[int]:
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        return ids or [1]  # 至少一个 token，避免全 padding

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        return cls(d["itos"])


def load_cosmetics_csv(path: str | Path, kb: KnowledgeBase, min_tokens: int = 3):
    """读取 Kaggle「Cosmetics datasets」格式的 CSV。

    需要列：Ingredients, Combination, Dry, Normal, Oily, Sensitive（0/1）。
    返回 [(tokens, labels)]。
    """
    samples = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("Ingredients") or "").strip()
            if not text or text.lower().startswith(("no info", "visit the", "#name")):
                continue
            toks = to_tokens(text, kb)
            if len(toks) < min_tokens:
                continue
            try:
                labels = [float(row[l]) for l in LABELS]
            except (KeyError, ValueError):
                continue
            samples.append((toks, labels))
    return samples


def load_jsonl(path: str | Path, kb: KnowledgeBase):
    """自建数据集：每行 {"ingredients": "...", "labels": {"Dry": 1, ...}}，缺省标签按 0。"""
    samples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        toks = to_tokens(d["ingredients"], kb)
        samples.append((toks, [float(d["labels"].get(l, 0)) for l in LABELS]))
    return samples


def make_synthetic(kb: KnowledgeBase, n: int = 3000, seed: int = 0, noise: float = 0.05):
    """生成合成数据，只用于跑通训练流程和单元测试，不代表真实规律。"""
    rng = random.Random(seed)
    ids = [i for i in kb.items if i != "water"]
    samples = []
    for _ in range(n):
        k = rng.randint(6, 20)
        toks = (["water"] if rng.random() < 0.85 else []) + rng.sample(ids, k)
        ings = [kb.items[t] for t in toks]
        top = ings[:8]
        irritating = any(i.irritation >= 2 for i in top) or any("fragrance" in i.tags for i in ings)
        comedo = any(i.comedogenic >= 4 for i in ings)
        rich = sum(f in ("occlusive", "emollient") for i in ings for f in i.functions)
        humect = sum("humectant" in i.functions for i in ings)
        y = {
            "Sensitive": not irritating,
            "Oily": not comedo and rich <= 2,
            "Dry": rich >= 2 or humect >= 3,
            "Normal": not (irritating and comedo),
            "Combination": not comedo and rich <= 3,
        }
        labels = [float(y[l]) for l in LABELS]
        labels = [1.0 - v if rng.random() < noise else v for v in labels]
        samples.append((toks, labels))
    return samples

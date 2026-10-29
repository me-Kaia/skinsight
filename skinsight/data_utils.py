"""数据处理：成分 token 化、词表、数据集加载、合成演示数据。不依赖 torch。"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from .kb import KnowledgeBase, normalize

LABELS = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
LABELS_ZH = ["混合肌", "干皮", "中性肌", "油皮", "敏感肌"]
CATEGORIES = ["Cleanser", "Eye cream", "Face Mask", "Moisturizer", "Sun protect", "Treatment"]  # CSV 的 Label 列
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


PLACEHOLDER_RE = re.compile(r"(?i)^\s*(no info|visit the|#name|this ingredient list is subject to change)")
DISCLAIMER_RE = re.compile(r"(?i)\s*(please be aware|this ingredient list is subject to change).*$", re.S)
HEADER_RE = re.compile(r"(?i)^.*?\b(?:inactive |active |other )?ingredients?\s*:")
WATER_RE = re.compile(r"(?i)\b(water|aqua|eau)\b\s*[,(/]")
PERCENT_RE = re.compile(r"\s*\(?\d+(?:\.\d+)?\s*%\)?")
FOOTNOTE_RE = re.compile(r"^\*")  # "*Organic." / "**Natural." 这类脚注；成分名后面的 * 由 normalize 去掉


def clean_ingredients(text: str) -> str | None:
    """清洗网页抓来的成分表；不是成分表（占位符、纯营销文案）时返回 None。

    处理：「-成分: 功效描述」营销文案、"...Ingredients:" 前缀、免责声明、
    "5.0%Water" 这类浓度粘连、"(and)" / "May Contain:" 分隔、"*Organic." 脚注。
    """
    text = (text or "").strip()
    if not text or PLACEHOLDER_RE.match(text):
        return None
    text = DISCLAIMER_RE.sub("", text)
    if text.startswith("-"):
        # 丝芙兰的「核心成分」营销文案，后面有时才接真正的成分表
        m = WATER_RE.search(text)
        if not m:
            return None
        text = text[m.start():]
    text = HEADER_RE.sub("", text, count=1)
    text = re.sub(r"%(?=[A-Za-z])", "%, ", text)
    text = PERCENT_RE.sub("", text)
    text = re.sub(r"(?i)\(and\)|\bmay contain\s*:?|\[?\+/-\]?", ",", text)
    text = re.sub(r"\.\s+(?=[A-Z*])", ", ", text)  # 句号分句 -> 分隔符，脚注单独成项
    text = re.sub(r"\s+(?=\*+\s*[A-Za-z])", ", ", text)  # "Phenoxyethanol * essential oil"
    parts = [p.strip() for p in re.split(r"(?<!\d),(?!\d)", text)]  # 同 kb.SPLIT_RE，不拆 1,2-己二醇
    # 只去掉逗号分出来的脚注项；中文顿号等分隔交给 kb.parse
    parts = [p for p in parts if p and not FOOTNOTE_RE.match(p)]
    return ", ".join(parts) or None


def kb_feature_names(kb: KnowledgeBase) -> list[str]:
    """知识库特征的维度名：刺激等级、致痘等级（各归一化到 0~1）+ 每个 tag 一个 one-hot。"""
    return ["irritation", "comedogenic", *sorted({t for i in kb.items.values() for t in i.tags})]


def kb_feature_matrix(vocab: Vocab, kb: KnowledgeBase) -> list[list[float]]:
    """[len(vocab), F]：知识库成分 token 取其特征，未识别成分（raw:*）、pad、unk 为零向量。"""
    names = kb_feature_names(kb)
    tag_idx = {t: k for k, t in enumerate(names)}
    rows = []
    for tok in vocab.itos:
        row = [0.0] * len(names)
        ing = kb.items.get(tok)
        if ing:
            row[0], row[1] = ing.irritation / 3, ing.comedogenic / 5
            for t in ing.tags:
                row[tag_idx[t]] = 1.0
        rows.append(row)
    return rows


def load_cosmetics_csv(path: str | Path, kb: KnowledgeBase, min_tokens: int = 3,
                       drop_unlabeled: bool = True, dedupe: bool = True, task: str = "skin"):
    """读取 Kaggle「Cosmetics datasets」格式的 CSV。

    需要列：Ingredients, Combination, Dry, Normal, Oily, Sensitive（0/1）；task="category" 时用 Label 列。
    task="skin"：labels 为 5 个肤质的 0/1；task="category"：labels 为 CATEGORIES 下标，
    此时不看肤质标签，drop_unlabeled 不起作用。
    drop_unlabeled：去掉 5 个标签全为 0 的行。这类行约占 1/3、在各品类里比例相近，
    更像是「未标注肤质」而不是「不适合任何肤质」，当负样本会引入噪声。
    dedupe：同一成分表只保留第一条（原数据同一产品会出现在多个品类里，否则会跨 train/test 泄漏）。
    返回 [(tokens, labels)]。
    """
    if task not in ("skin", "category"):
        raise ValueError(f"未知 task: {task}")
    samples, seen = [], set()
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = clean_ingredients(row.get("Ingredients"))
            if text is None:
                continue
            if task == "category":
                if row.get("Label") not in CATEGORIES:
                    continue
                labels = CATEGORIES.index(row["Label"])
            else:
                try:
                    labels = [float(row[l]) for l in LABELS]
                except (KeyError, ValueError):
                    continue
                if drop_unlabeled and not any(labels):
                    continue
            toks = to_tokens(text, kb)
            if len(toks) < min_tokens:
                continue
            key = tuple(toks)
            if dedupe and key in seen:
                continue
            seen.add(key)
            samples.append((toks, labels))
    return samples


def split_samples(samples, seed: int = 42, frac: float = 0.1):
    """固定种子打乱后切成 train/val/test（val、test 各占 frac）。模型和 baseline 共用。"""
    samples = list(samples)
    random.Random(seed).shuffle(samples)
    n = len(samples); k = max(1, int(n * frac))
    return samples[2 * k:], samples[k:2 * k], samples[:k]


def f1_scores(y_true, y_pred) -> list[float]:
    """逐标签 F1，y_* 为 [[0/1] * len(LABELS)]；与 train.evaluate 同一口径（无正样本时记 0）。"""
    out = []
    for j in range(len(LABELS)):
        tp = sum(t[j] and p[j] for t, p in zip(y_true, y_pred))
        fp = sum((not t[j]) and p[j] for t, p in zip(y_true, y_pred))
        fn = sum(t[j] and not p[j] for t, p in zip(y_true, y_pred))
        out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return out


def multiclass_scores(y_true: list[int], y_pred: list[int], n_classes: int) -> dict:
    """多分类：accuracy、逐类 F1、macro-F1。"""
    f1 = []
    for c in range(n_classes):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    return {"acc": acc, "f1": f1, "macro_f1": sum(f1) / n_classes}


def auc_scores(y_true, scores) -> list[float | None]:
    """逐标签 ROC-AUC（秩统计，并列取平均秩）；某标签只有一类时为 None。与阈值无关，全预测为正时是 0.5。"""
    out = []
    for j in range(len(LABELS)):
        pairs = sorted((s[j], t[j]) for s, t in zip(scores, y_true))
        n_pos = sum(t for _, t in pairs); n_neg = len(pairs) - n_pos
        if not n_pos or not n_neg:
            out.append(None); continue
        rank_sum, i = 0.0, 0
        while i < len(pairs):
            k = i
            while k < len(pairs) and pairs[k][0] == pairs[i][0]:
                k += 1
            rank_sum += sum(t for _, t in pairs[i:k]) * (i + k + 1) / 2  # 秩从 1 开始
            i = k
        out.append((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return out


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

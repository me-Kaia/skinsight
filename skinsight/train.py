"""训练肤质适配模型。

    python -m skinsight.train --synthetic                       # 用合成数据跑通流程
    python -m skinsight.train --csv data/cosmetics.csv          # Kaggle Cosmetics datasets
    python -m skinsight.baseline --csv data/cosmetics.csv       # 同划分的词袋 + 逻辑回归对比
    python -m skinsight.train --jsonl data/my_products.jsonl    # 自建数据
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data_utils import (LABELS, Vocab, auc_scores, load_cosmetics_csv, load_jsonl, make_synthetic,
                         split_samples)
from .kb import KnowledgeBase
from .model import SkinTypeNet


class SkinDataset(Dataset):
    def __init__(self, samples, vocab: Vocab, max_len: int):
        self.data = [(vocab.encode(t, max_len), y) for t, y in samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    L = max(len(ids) for ids, _ in batch)
    x = torch.zeros(len(batch), L, dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
    y = torch.tensor([lab for _, lab in batch], dtype=torch.float)
    return x, y


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = torch.zeros(len(LABELS)); fp = torch.zeros(len(LABELS)); fn = torch.zeros(len(LABELS))
    correct = torch.zeros(len(LABELS)); total = 0
    probs, ys = [], []
    for x, y in loader:
        logits, _ = model(x.to(device))
        prob = torch.sigmoid(logits).cpu()
        probs += prob.tolist(); ys += y.tolist()
        pred = (prob > 0.5).float()
        tp += (pred * y).sum(0); fp += (pred * (1 - y)).sum(0); fn += ((1 - pred) * y).sum(0)
        correct += (pred == y).float().sum(0); total += y.size(0)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    return {"acc": (correct / max(total, 1)).tolist(), "f1": f1.tolist(), "macro_f1": f1.mean().item(),
            "auc": auc_scores(ys, probs)}


def fmt(m):
    per = "  ".join(f"{l}:{f:.2f}" for l, f in zip(LABELS, m["f1"]))
    return f"macro-F1 {m['macro_f1']:.3f} | {per}"


def main(argv=None):
    # Windows 中文系统重定向/管道输出时默认 GBK，与 cli.py 保持一致统一用 utf-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv"); src.add_argument("--jsonl"); src.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-unlabeled", action="store_true", help="CSV：保留 5 个标签全 0 的行（默认丢弃）")
    p.add_argument("--out", default="checkpoints/skin_type.pt")
    p.add_argument("--metrics-out", help="把 test 指标写成 JSON（便于和 baseline 对比）")
    args = p.parse_args(argv)

    random.seed(args.seed); torch.manual_seed(args.seed)
    kb = KnowledgeBase()
    if args.csv:
        samples = load_cosmetics_csv(args.csv, kb, drop_unlabeled=not args.keep_unlabeled)
    elif args.jsonl:
        samples = load_jsonl(args.jsonl, kb)
    else:
        samples = make_synthetic(kb, seed=args.seed)
    if len(samples) < 50:
        raise SystemExit(f"样本太少（{len(samples)} 条），检查数据格式。")

    train, val, test = split_samples(samples, seed=args.seed)

    # 知识库里的成分始终进词表，这样中文成分表在推理时也能对上
    vocab = Vocab.build([t for t, _ in train], min_freq=args.min_freq, always=list(kb.items))
    print(f"样本 train/val/test = {len(train)}/{len(val)}/{len(test)}，词表 {len(vocab)}")

    mk = lambda s, sh: DataLoader(SkinDataset(s, vocab, args.max_len), batch_size=args.batch_size,
                                  shuffle=sh, collate_fn=collate)
    train_dl, val_dl, test_dl = mk(train, True), mk(val, False), mk(test, False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkinTypeNet(len(vocab), num_labels=len(LABELS), d_model=args.d_model,
                        num_layers=args.layers, max_len=args.max_len, dropout=args.dropout).to(device)

    ys = torch.tensor([y for _, y in train])
    pos = ys.sum(0).clamp(min=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=((len(ys) - pos) / pos).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item() * x.size(0)
        m = evaluate(model, val_dl, device)
        print(f"epoch {epoch:3d} | loss {total / len(train):.4f} | val {fmt(m)}")
        if m["macro_f1"] > best:
            best, best_epoch = m["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_dl, device)
    print(f"best epoch {best_epoch} (val macro-F1 {best:.3f})")
    print("test", fmt(test_m))
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(
            {"model": "SkinTypeNet", "args": vars(args), "best_epoch": best_epoch, "val_macro_f1": best,
             "n": [len(train), len(val), len(test)], "test": test_m}, ensure_ascii=False, indent=2), encoding="utf-8")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": model.config, "vocab": vocab.to_dict(),
                "labels": LABELS, "synthetic": bool(args.synthetic)}, args.out)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()

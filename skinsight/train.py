"""训练肤质适配模型。

    python -m skinsight.train --synthetic                       # 用合成数据跑通流程
    python -m skinsight.train --csv data/cosmetics.csv          # Kaggle Cosmetics datasets
    python -m skinsight.baseline --csv data/cosmetics.csv       # 同划分的词袋 + 逻辑回归对比
    python -m skinsight.train --csv data/cosmetics.csv --kb-feats            # 加知识库特征
    python -m skinsight.train --csv data/cosmetics.csv --task category       # 产品类别多分类
    python -m skinsight.train --csv data/cosmetics.csv --init-ckpt checkpoints/pretrained.pt [--freeze-encoder]
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

from .data_utils import (CATEGORIES, LABELS, Vocab, auc_scores, kb_feature_matrix, load_cosmetics_csv,
                         load_jsonl, make_synthetic, multiclass_scores, split_samples)
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
    labs = [lab for _, lab in batch]
    # 多标签（肤质）为 0/1 列表 -> float；多分类（类别）为类下标 -> long
    y = torch.tensor(labs, dtype=torch.long if isinstance(labs[0], int) else torch.float)
    return x, y


@torch.no_grad()
def evaluate_category(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        logits, _ = model(x.to(device))
        preds += logits.argmax(-1).cpu().tolist(); ys += y.tolist()
    return multiclass_scores(ys, preds, len(CATEGORIES))


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


def fmt(m, names=LABELS):
    per = "  ".join(f"{l}:{f:.2f}" for l, f in zip(names, m["f1"]))
    acc = f"acc {m['acc']:.3f} | " if isinstance(m["acc"], float) else ""  # 多分类才有单个 accuracy
    return f"{acc}macro-F1 {m['macro_f1']:.3f} | {per}"


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
    p.add_argument("--task", choices=["skin", "category"], default="skin",
                   help="skin：5 种肤质多标签；category：产品类别多分类（只支持 --csv）")
    p.add_argument("--kb-feats", action="store_true", help="把知识库的刺激/致痘等级和 tags 加到 token embedding 上")
    p.add_argument("--init-ckpt", help="pretrain.py 的 checkpoint：沿用其词表和结构，加载 encoder 权重后微调")
    p.add_argument("--freeze-encoder", action="store_true", help="配合 --init-ckpt：冻结 encoder，只训练注意力池化和分类头")
    p.add_argument("--out", help="默认 checkpoints/skin_type.pt（category 任务为 checkpoints/category.pt）")
    p.add_argument("--metrics-out", help="把 test 指标写成 JSON（便于和 baseline 对比）")
    args = p.parse_args(argv)
    if args.freeze_encoder and not args.init_ckpt:
        p.error("--freeze-encoder 需要 --init-ckpt")
    if args.task == "category" and not args.csv:
        p.error("--task category 需要 --csv（类别来自 CSV 的 Label 列）")
    args.out = args.out or ("checkpoints/category.pt" if args.task == "category" else "checkpoints/skin_type.pt")
    is_cat = args.task == "category"
    label_names = CATEGORIES if is_cat else LABELS
    evaluate_fn = evaluate_category if is_cat else evaluate

    random.seed(args.seed); torch.manual_seed(args.seed)
    kb = KnowledgeBase()
    if args.csv:
        samples = load_cosmetics_csv(args.csv, kb, drop_unlabeled=not args.keep_unlabeled, task=args.task)
    elif args.jsonl:
        samples = load_jsonl(args.jsonl, kb)
    else:
        samples = make_synthetic(kb, seed=args.seed)
    if len(samples) < 50:
        raise SystemExit(f"样本太少（{len(samples)} 条），检查数据格式。")

    train, val, test = split_samples(samples, seed=args.seed)

    # 知识库里的成分始终进词表，这样中文成分表在推理时也能对上
    pre = None
    if args.init_ckpt:
        pre = torch.load(args.init_ckpt, map_location="cpu")
        vocab = Vocab.from_dict(pre["vocab"])  # 预训练词表，已包含知识库成分
        for k in ("d_model", "num_layers", "max_len", "dropout"):
            setattr(args, {"num_layers": "layers"}.get(k, k), pre["config"][k])
    else:
        vocab = Vocab.build([t for t, _ in train], min_freq=args.min_freq, always=list(kb.items))
    print(f"样本 train/val/test = {len(train)}/{len(val)}/{len(test)}，词表 {len(vocab)}")

    mk = lambda s, sh: DataLoader(SkinDataset(s, vocab, args.max_len), batch_size=args.batch_size,
                                  shuffle=sh, collate_fn=collate)
    train_dl, val_dl, test_dl = mk(train, True), mk(val, False), mk(test, False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kb_feats = kb_feature_matrix(vocab, kb) if args.kb_feats else None
    model = SkinTypeNet(len(vocab), num_labels=len(label_names), d_model=args.d_model,
                        nhead=pre["config"]["nhead"] if pre else 4,
                        num_layers=args.layers, max_len=args.max_len, dropout=args.dropout,
                        kb_feat_dim=len(kb_feats[0]) if kb_feats else 0)
    if pre:
        model.load_encoder(pre["model"])
        if args.freeze_encoder:
            for prm in model.encoder_parameters():
                prm.requires_grad_(False)
    if kb_feats:
        model.kb_feats.copy_(torch.tensor(kb_feats))
    model = model.to(device)

    ys = torch.tensor([y for _, y in train])
    if is_cat:
        loss_fn = nn.CrossEntropyLoss()  # 6 个类别 170~298 条，比较均衡，不加权
    else:
        pos = ys.sum(0).clamp(min=1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=((len(ys) - pos) / pos).to(device))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=1e-2)

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
        m = evaluate_fn(model, val_dl, device)
        print(f"epoch {epoch:3d} | loss {total / len(train):.4f} | val {fmt(m, label_names)}")
        if m["macro_f1"] > best:
            best, best_epoch = m["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_m = evaluate_fn(model, test_dl, device)
    print(f"best epoch {best_epoch} (val macro-F1 {best:.3f})")
    print("test", fmt(test_m, label_names))
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(
            {"model": "SkinTypeNet", "args": vars(args), "best_epoch": best_epoch, "val_macro_f1": best,
             "n": [len(train), len(val), len(test)], "test": test_m}, ensure_ascii=False, indent=2), encoding="utf-8")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": model.config, "vocab": vocab.to_dict(),
                "labels": label_names, "task": args.task, "synthetic": bool(args.synthetic)}, args.out)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()

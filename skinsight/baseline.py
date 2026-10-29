"""对比用 baseline：成分词袋（0/1）+ 每个标签一个逻辑回归。

和 train.py 用同一套数据加载、同一个 split_samples 划分和同一 F1 口径：

    python -m skinsight.baseline --csv data/cosmetics.csv
    python -m skinsight.baseline --synthetic

另外输出「全部预测为适合」的平凡基线，标签正样本比例很高时它的 F1 并不低，可作参照。
需要 scikit-learn。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from .data_utils import (LABELS, auc_scores, f1_scores, load_cosmetics_csv, load_jsonl, make_synthetic,
                         split_samples)
from .kb import KnowledgeBase


def fmt(f1):
    per = "  ".join(f"{l}:{f:.2f}" for l, f in zip(LABELS, f1))
    return f"macro-F1 {sum(f1) / len(f1):.3f} | {per}"


def fit_predict(train, evals, C: float, min_df: int):
    """每个标签单独一个 class_weight=balanced 的逻辑回归（对应模型里的 pos_weight）。"""
    vec = CountVectorizer(analyzer=lambda toks: toks, binary=True, min_df=min_df)
    X = vec.fit_transform([t for t, _ in train])
    Xs = [vec.transform([t for t, _ in s]) for s in evals]
    preds = [[[0] * len(LABELS) for _ in s] for s in evals]
    probs = [[[0.0] * len(LABELS) for _ in s] for s in evals]
    for j in range(len(LABELS)):
        y = [int(lab[j]) for _, lab in train]
        if len(set(y)) < 2:  # 训练集里全是同一类，直接预测这一类
            for p, q in zip(preds, probs):
                for row, prow in zip(p, q):
                    row[j] = y[0]; prow[j] = float(y[0])
            continue
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X, y)
        for p, q, Xe in zip(preds, probs, Xs):
            for row, prow, v, pr in zip(p, q, clf.predict(Xe), clf.predict_proba(Xe)[:, 1]):
                row[j] = int(v); prow[j] = float(pr)
    return preds, probs


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv"); src.add_argument("--jsonl"); src.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-freq", type=int, default=2, help="同 train.py 的词表最小词频")
    p.add_argument("--keep-unlabeled", action="store_true")
    p.add_argument("--metrics-out")
    args = p.parse_args(argv)

    kb = KnowledgeBase()
    if args.csv:
        samples = load_cosmetics_csv(args.csv, kb, drop_unlabeled=not args.keep_unlabeled)
    elif args.jsonl:
        samples = load_jsonl(args.jsonl, kb)
    else:
        samples = make_synthetic(kb, seed=args.seed)
    train, val, test = split_samples(samples, seed=args.seed)
    print(f"样本 train/val/test = {len(train)}/{len(val)}/{len(test)}")

    y_val, y_test = [y for _, y in val], [y for _, y in test]
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        (pv,), _ = fit_predict(train, [val], C, args.min_freq)
        f = f1_scores(y_val, pv)
        print(f"C={C:<5} val {fmt(f)}")
        if best is None or sum(f) > sum(best[1]):
            best = (C, f)
    C = best[0]
    (pt,), (qt,) = fit_predict(train, [test], C, args.min_freq)
    test_f1 = f1_scores(y_test, pt)
    test_auc = auc_scores(y_test, qt)
    print(f"best C={C}")
    print("test logreg     ", fmt(test_f1))
    allpos_f1 = f1_scores(y_test, [[1] * len(LABELS) for _ in test])
    print("test all-positive", fmt(allpos_f1))

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(
            {"model": "logreg-bow", "args": vars(args), "C": C, "val_macro_f1": sum(best[1]) / len(LABELS),
             "n": [len(train), len(val), len(test)],
             "test": {"f1": test_f1, "macro_f1": sum(test_f1) / len(LABELS), "auc": test_auc},
             "all_positive": {"f1": allpos_f1, "macro_f1": sum(allpos_f1) / len(LABELS)}},
            ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

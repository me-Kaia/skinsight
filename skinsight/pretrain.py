"""自监督预训练：在无标签成分表上做 masked ingredient modeling。

    python -m skinsight.pretrain --corpus data/openbeautyfacts/pretrain.jsonl \\
        --cosmetics-csv data/cosmetics.csv --epochs 20

随机选 15% 的成分（pad/unk 不选）作为预测目标，其中 80% 换成 [MASK]、10% 换成随机成分、
10% 保持不变（BERT 的做法），用 encoder 输出预测原成分。留 val-frac 的语料做验证，
报告被遮成分的 top-1 / top-5 准确率，并与「总猜最常见的成分」的频率基线对比。

词表 = 预训练语料中出现 ≥ min_freq 次的成分 ∪ cosmetics.csv 成分表（只用文本、不用标签）中
出现 ≥ min_freq 次的成分 ∪ 知识库成分。下游微调用 train.py --init-ckpt，沿用这个词表。
MLM 只在预训练语料上做，不看 cosmetics.csv 的成分表。
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_utils import MASK, PAD, UNK, Vocab, clean_ingredients, to_tokens
from .kb import KnowledgeBase
from .model import MaskedIngredientModel
from .pretrain_data import load_corpus_tokens

IGNORE = -100


def build_vocab(corpus: list[list[str]], extra_lists: list[list[str]], kb: KnowledgeBase, min_freq: int = 2) -> Vocab:
    keep = set()
    for lists in (corpus, extra_lists):
        cnt = Counter(t for toks in lists for t in toks)
        keep |= {t for t, c in cnt.items() if c >= min_freq}
    keep |= set(kb.items)
    return Vocab([PAD, UNK, MASK, *sorted(keep)])


def cosmetics_lists(path: str | Path, kb: KnowledgeBase) -> list[list[str]]:
    import csv
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = clean_ingredients(row.get("Ingredients"))
            if text:
                out.append(to_tokens(text, kb))
    return out


def mask_tokens(x: torch.Tensor, mask_id: int, vocab_size: int, p: float = 0.15,
                gen: torch.Generator | None = None):
    """x: [B, L]。返回 (输入, 目标)；目标里不预测的位置为 IGNORE。每行至少遮 1 个（若有可遮的）。"""
    cand = x > 2  # 跳过 pad=0 / unk=1 / mask=2
    r = torch.rand(x.shape, generator=gen)
    sel = cand & (r < p)
    # 一个都没选中的行，强制选分数最小的候选位置
    none = ~sel.any(1) & cand.any(1)
    if none.any():
        forced = r.masked_fill(~cand, 2.0).argmin(1)
        sel[none, forced[none]] = True
    targets = torch.where(sel, x, torch.full_like(x, IGNORE))
    action = torch.rand(x.shape, generator=gen)
    inp = x.clone()
    inp[sel & (action < 0.8)] = mask_id
    rand_pos = sel & (action >= 0.8) & (action < 0.9)
    inp[rand_pos] = torch.randint(3, vocab_size, (int(rand_pos.sum()),), generator=gen)
    return inp, targets


def batches(seqs: list[list[int]], bs: int, shuffle: bool, rng: random.Random | None = None):
    order = list(range(len(seqs)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        chunk = [seqs[j] for j in order[i:i + bs]]
        L = max(map(len, chunk))
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        for k, s in enumerate(chunk):
            x[k, :len(s)] = torch.tensor(s)
        yield x


@torch.no_grad()
def evaluate(model, val_seqs, mask_id, vocab_size, bs, unigram_top5: torch.Tensor):
    """固定随机种子遮罩，返回模型和频率基线的 top-1 / top-5 准确率、MLM loss。"""
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    n = hit1 = hit5 = base1 = base5 = 0; loss_sum = 0.0
    for x in batches(val_seqs, bs, shuffle=False):
        inp, tgt = mask_tokens(x, mask_id, vocab_size, gen=gen)
        logits = model(inp)
        m = tgt != IGNORE
        lg, t = logits[m], tgt[m]
        loss_sum += F.cross_entropy(lg, t, reduction="sum").item()
        top5 = lg.topk(5, -1).indices
        hit1 += (top5[:, 0] == t).sum().item(); hit5 += (top5 == t[:, None]).any(1).sum().item()
        base1 += (t == unigram_top5[0]).sum().item(); base5 += torch.isin(t, unigram_top5).sum().item()
        n += t.numel()
    return {"n_masked": n, "loss": loss_sum / n, "top1": hit1 / n, "top5": hit5 / n,
            "unigram_top1": base1 / n, "unigram_top5": base5 / n}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="pretrain_data.py 生成的 jsonl")
    p.add_argument("--cosmetics-csv", help="下游数据，只用来扩充词表（不用标签、不参与 MLM）")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=float, default=0.05, help="warmup 步数占总步数比例")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--time-limit", type=float, help="分钟；到时间就在当前 epoch 结束后停止")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/pretrained.pt")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    kb = KnowledgeBase()
    corpus = load_corpus_tokens(args.corpus, kb)
    rng.shuffle(corpus)
    n_val = max(1, int(len(corpus) * args.val_frac))
    val_toks, train_toks = corpus[:n_val], corpus[n_val:]
    extra = cosmetics_lists(args.cosmetics_csv, kb) if args.cosmetics_csv else []
    # 词表只看预训练训练部分，验证集里的新成分记为 unk、不作为预测目标
    vocab = build_vocab(train_toks, extra, kb, args.min_freq)
    mask_id = vocab.stoi[MASK]
    enc = lambda ts: [vocab.encode(t, args.max_len) for t in ts]
    train_seqs, val_seqs = enc(train_toks), enc(val_toks)
    freq = Counter(i for s in train_seqs for i in s if i > 2)
    unigram_top5 = torch.tensor([i for i, _ in freq.most_common(5)])
    unk_rate = sum(i == 1 for s in train_seqs for i in s) / sum(map(len, train_seqs))
    print(f"语料 train/val = {len(train_seqs)}/{len(val_seqs)}，词表 {len(vocab)}，训练集 unk 比例 {unk_rate:.3f}")

    model = MaskedIngredientModel(vocab_size=len(vocab), d_model=args.d_model, num_layers=args.layers,
                                  max_len=args.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = math.ceil(len(train_seqs) / args.batch_size)
    total = steps_per_epoch * args.epochs; warm = max(1, int(total * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))

    t0 = time.time(); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0; n = 0
        for x in batches(train_seqs, args.batch_size, shuffle=True, rng=rng):
            inp, tgt = mask_tokens(x, mask_id, len(vocab))
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=IGNORE)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            loss_sum += loss.item() * x.size(0); n += x.size(0)
        m = evaluate(model, val_seqs, mask_id, len(vocab), 256, unigram_top5)
        elapsed = (time.time() - t0) / 60
        history.append({"epoch": epoch, "train_loss": loss_sum / n, **m, "minutes": elapsed})
        print(f"epoch {epoch:3d} | train loss {loss_sum / n:.3f} | val loss {m['loss']:.3f} "
              f"top1 {m['top1']:.3f} top5 {m['top5']:.3f} | {elapsed:.1f} min")
        if args.time_limit and elapsed > args.time_limit:
            print(f"超过时间上限 {args.time_limit} 分钟，停止"); break

    m = history[-1]
    print(f"val masked top-1 {m['top1']:.3f} / top-5 {m['top5']:.3f}（频率基线 {m['unigram_top1']:.3f} / "
          f"{m['unigram_top5']:.3f}，{m['n_masked']} 个被遮成分）")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.net.state_dict(), "config": model.net.config, "vocab": vocab.to_dict(),
                "mlm": model.state_dict(), "args": vars(args), "history": history,
                "task": "pretrain"}, args.out)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()

"""准备自监督预训练语料：Open Beauty Facts 的无标签成分表。

    curl -L -o data/openbeautyfacts/en.openbeautyfacts.org.products.csv.gz \\
        https://static.openbeautyfacts.org/data/en.openbeautyfacts.org.products.csv.gz
    python -m skinsight.pretrain_data --obf data/openbeautyfacts/en.openbeautyfacts.org.products.csv.gz \\
        --exclude-csv data/cosmetics.csv --out data/openbeautyfacts/pretrain.jsonl

数据来自 Open Beauty Facts（https://world.openbeautyfacts.org），ODbL 协议，不随仓库分发。
处理：取 ingredients_text 非空 -> clean_ingredients -> 至少 5 个成分 -> 按 token 序列去重 ->
去掉与 cosmetics.csv 任意一条（清洗后）token 序列完全相同的条目，避免下游评估泄漏。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

from .data_utils import clean_ingredients, to_tokens
from .kb import KnowledgeBase


def cosmetics_token_set(path: str | Path, kb: KnowledgeBase) -> set[tuple[str, ...]]:
    """cosmetics.csv 所有行（不论标签）清洗后的 token 序列。"""
    out = set()
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = clean_ingredients(row.get("Ingredients"))
            if text:
                out.add(tuple(to_tokens(text, kb)))
    return out


def iter_obf(path: str | Path):
    """逐行读 OBF 的 CSV 导出（tab 分隔，可为 .gz），产出 (code, ingredients_text)。"""
    csv.field_size_limit(1 << 24)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            yield row.get("code") or "", (row.get("ingredients_text") or "").strip()


def build_corpus(obf_path, kb: KnowledgeBase, exclude: set | None = None, min_tokens: int = 5):
    stats = Counter()
    seen, corpus = set(), []
    for code, raw in iter_obf(obf_path):
        stats["rows"] += 1
        if not raw:
            continue
        stats["has_text"] += 1
        text = clean_ingredients(raw)
        if text is None:
            stats["drop_not_ingredients"] += 1; continue
        toks = to_tokens(text, kb)
        if len(toks) < min_tokens:
            stats["drop_short"] += 1; continue
        key = tuple(toks)
        if key in seen:
            stats["drop_dup"] += 1; continue
        seen.add(key)
        if exclude and key in exclude:
            stats["drop_in_cosmetics"] += 1; continue
        corpus.append({"code": code, "ingredients": text})
    stats["kept"] = len(corpus)
    return corpus, stats


def load_corpus_tokens(path: str | Path, kb: KnowledgeBase) -> list[list[str]]:
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    return [to_tokens(json.loads(r)["ingredients"], kb) for r in rows if r.strip()]


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--obf", required=True, help="Open Beauty Facts CSV 导出（.csv 或 .csv.gz）")
    p.add_argument("--exclude-csv", help="cosmetics.csv，与其成分表完全相同的条目会被去掉")
    p.add_argument("--min-tokens", type=int, default=5)
    p.add_argument("--out", default="data/openbeautyfacts/pretrain.jsonl")
    args = p.parse_args(argv)

    kb = KnowledgeBase()
    exclude = cosmetics_token_set(args.exclude_csv, kb) if args.exclude_csv else None
    corpus, stats = build_corpus(args.obf, kb, exclude, args.min_tokens)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in stats.items():
        print(f"{k:22s} {v}")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()

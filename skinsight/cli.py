"""命令行入口。

    python -m skinsight analyze "水, 甘油, 烟酰胺, 香精"
    python -m skinsight analyze "..." --ckpt checkpoints/skin_type.pt
    python -m skinsight check -p "A醇精华: 水, 视黄醇, ..." -p "水杨酸水: 水, 水杨酸, ..."
"""
from __future__ import annotations

import argparse
import sys

from .conflicts import LEVEL_ZH, check_routine
from .kb import KnowledgeBase, summarize_flags

ICON = {"warn": "⚠️ ", "note": "ℹ️ "}


def cmd_analyze(args, kb):
    text = args.text if args.text != "-" else sys.stdin.read()
    items = kb.parse(text)
    print(f"共 {len(items)} 个成分，识别 {sum(1 for i in items if i.ingredient)} 个：\n")
    for it in items:
        if it.ingredient:
            ing = it.ingredient
            mark = "（模糊匹配）" if it.match == "fuzzy" else ""
            func = "/".join(ing.functions) or "-"
            print(f"  {it.position + 1:>2}. {ing.display}  {ing.inci}{mark}  [{func}]")
        else:
            print(f"  {it.position + 1:>2}. {it.raw}  [未收录]")
    flags = summarize_flags(items)
    if flags:
        print("\n提示：")
        for level, msg in flags:
            print(f"  {ICON[level]}{msg}")
    if args.ckpt:
        from .predict import Predictor  # 只有需要时才导入 torch
        try:
            pred = Predictor(args.ckpt, kb)
        except ValueError as e:
            raise SystemExit(f"错误：{e}")
        probs, top = pred.predict(text)
        print("\n肤质适配度（模型预测）：")
        if pred.synthetic:
            print("  ⚠️ 当前模型由合成数据训练，结果仅用于演示。")
        for name, p in probs.items():
            print(f"  {name:<4} {p:5.1%} " + "█" * round(p * 20))
        print("  影响最大的成分：" + "、".join(f"{n}({w:.2f})" for n, w in top))


def cmd_check(args, kb):
    products = []
    for i, spec in enumerate(args.product, 1):
        name, sep, text = spec.partition(":")
        if not sep:
            name, sep, text = spec.partition("：")
        if not sep:
            name, text = f"产品{i}", spec
        products.append((name.strip(), kb.parse(text)))
    findings = check_routine(products)
    if not findings:
        print("没有发现已知的叠用冲突。")
        return
    for f in findings:
        print(f"[{LEVEL_ZH[f.level]}] {' + '.join(f.products)}：{f.message}")


def main(argv=None):
    # Windows 中文系统重定向/管道输出时默认 GBK，无法编码 ⚠️ 等 emoji
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="skinsight", description="护肤品成分表解读 / 叠用冲突检查")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="解读一个成分表")
    a.add_argument("text", help="成分表文本，传 - 则从标准输入读取")
    a.add_argument("--ckpt", help="模型 checkpoint 路径（可选）")
    c = sub.add_parser("check", help="检查多个产品叠用冲突")
    c.add_argument("-p", "--product", action="append", required=True, help="'产品名: 成分表'，可重复")
    args = p.parse_args(argv)
    kb = KnowledgeBase()
    {"analyze": cmd_analyze, "check": cmd_check}[args.cmd](args, kb)


if __name__ == "__main__":
    main()

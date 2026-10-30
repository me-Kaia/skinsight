"""把魔搭创空间需要的文件复制到一个目录（通常是 git clone 下来的创空间仓库）。

    python deploy/modelscope/prepare.py <创空间仓库目录>

只复制网页 demo 用到的部分：app.py、知识库和规则引擎（不含模型、训练代码和数据）。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FILES = {
    "app.py": ROOT / "app.py",
    "LICENSE": ROOT / "LICENSE",
    "README.md": HERE / "README.md",
    "requirements.txt": HERE / "requirements.txt",
    "skinsight/__init__.py": ROOT / "skinsight/__init__.py",
    "skinsight/kb.py": ROOT / "skinsight/kb.py",
    "skinsight/conflicts.py": ROOT / "skinsight/conflicts.py",
    "skinsight/data/ingredients.json": ROOT / "skinsight/data/ingredients.json",
}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(__doc__)
    out = Path(argv[0])
    if out.resolve() == ROOT:
        raise SystemExit("目标目录不能是项目根目录")
    for rel, src in FILES.items():
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"{rel}")
    print(f"已复制 {len(FILES)} 个文件到 {out}")


if __name__ == "__main__":
    main()

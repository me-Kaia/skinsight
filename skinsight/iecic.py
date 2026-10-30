"""《已使用化妆品原料目录（2021年版）》备用字典。

知识库（kb.py）只收录了几百个常见成分；没收录的成分用这份目录确认「是不是登记过的化妆品原料」，
疑似 OCR 错字时给出目录里最接近的名字供用户确认（不自动替换）。
数据来自国家药监局公告附件，只保留序号、标准中文名、INCI 名，见 data/iecic_2021.tsv 文件头。
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from .kb import PAREN_RE, normalize

DEFAULT_PATH = Path(__file__).parent / "data" / "iecic_2021.tsv"
_COMPACT_RE = re.compile(r"[\s/.,\-]+")
_CJK_RE = re.compile(r"[一-鿿]")


def _compact(s: str) -> str:
    return _COMPACT_RE.sub("", normalize(s))


@dataclass(frozen=True)
class CatalogEntry:
    no: int
    zh: str
    inci: str


class Catalog:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.entries: list[CatalogEntry] = []
        self.index: dict[str, CatalogEntry] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("no\t"):
                continue
            no, zh, inci = (line.split("\t") + ["", ""])[:3]
            e = CatalogEntry(int(no), zh.rstrip("*"), inci)
            self.entries.append(e)
            keys = [e.zh, PAREN_RE.sub("", e.zh)]  # 「牛油果树（BUTYROSPERMUM PARKII）果脂」也能用「牛油果树果脂」查
            if inci:
                keys.append(inci)
            for k in keys:
                self.index.setdefault(_compact(k), e)
        self._zh_names = sorted({e.zh for e in self.entries})
        self._inci_names = sorted({e.inci for e in self.entries if e.inci})
        self._suggest_cache: dict[str, list[str]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def lookup(self, name: str) -> CatalogEntry | None:
        """名称精确对上目录（忽略大小写、空格、标点；中文名可以省略括号里的拉丁名）。"""
        name = name.strip()
        if not name:
            return None
        return self.index.get(_compact(name)) or self.index.get(_compact(PAREN_RE.sub("", name)))

    def suggest(self, name: str, n: int = 3, kb=None) -> list[str]:
        """疑似错字时，目录里最接近的名字（最多 n 个）。

        除了字形相近的名字，还检查是不是漏了分隔符、两个成分粘在一起（"甘草酸二钾丙烯酸（酯）类…"），
        是的话给出「A、B」的拆分建议。
        """
        name = name.strip()
        if name in self._suggest_cache:
            return self._suggest_cache[name]
        known = lambda s: (kb is not None and kb.lookup(s)[1] in ("exact", "paren")) or self.lookup(s) is not None
        out = []
        cjk = bool(_CJK_RE.search(name))
        if cjk:
            for i in range(2, len(name) - 1):
                a, b = name[:i].strip(), name[i:].strip()
                if known(a) and known(b):
                    out.append(f"{a}、{b}")
                    break
        pool = self._zh_names if cjk else self._inci_names
        key = name if cjk else name.upper()
        cutoff = 0.66 if cjk else 0.8  # 3 个字错 1 个是 0.67
        cands = difflib.get_close_matches(key, pool, n=20, cutoff=cutoff)

        k2 = key.replace(" ", "")

        def score(m):
            # OCR 认错字通常不改变字数（不算空格）；知识库收录的常见成分优先
            m2 = m.replace(" ", "")
            r = difflib.SequenceMatcher(None, k2, m2).ratio()
            return r + 0.1 * (len(m2) == len(k2)) + 0.05 * bool(kb is not None and kb.lookup(m)[1] == "exact")
        for m in sorted(cands, key=score, reverse=True):
            if m not in out:
                out.append(m)
        self._suggest_cache[name] = out = out[:n]
        return out

"""成分知识库：加载、别名索引、成分表解析与匹配。"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_KB = Path(__file__).parent / "data" / "ingredients.json"

# 按逗号/顿号/分号/换行切分；数字之间的逗号（如 1,2-己二醇）不切
SPLIT_RE = re.compile(r"(?<!\d),(?!\d)|[，、;；\n]+")
PAREN_RE = re.compile(r"[(（]([^)）]*)[)）]")


def normalize(s: str) -> str:
    """全角转半角、小写、去首尾标点与多余空格。"""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .。*†‡")


@dataclass
class Ingredient:
    id: str
    inci: str
    zh: str
    aliases: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    irritation: int = 0
    comedogenic: int = 0
    note: str = ""


@dataclass
class ParsedItem:
    raw: str
    position: int                     # 在成分表中的位置（0 起）；越靠前通常浓度越高
    ingredient: Optional[Ingredient]  # 未识别时为 None
    match: str                        # exact / paren / fuzzy / none


class KnowledgeBase:
    def __init__(self, path: Path | str = DEFAULT_KB, fuzzy_cutoff: float = 0.88):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.fuzzy_cutoff = fuzzy_cutoff
        self.items: dict[str, Ingredient] = {}
        self.index: dict[str, str] = {}
        for d in data["ingredients"]:
            ing = Ingredient(**d)
            if ing.id in self.items:
                raise ValueError(f"重复的成分 id: {ing.id}")
            self.items[ing.id] = ing
            for key in [ing.inci, ing.zh, *ing.aliases]:
                k = normalize(key)
                if k in self.index and self.index[k] != ing.id:
                    raise ValueError(f"别名冲突: {key!r} -> {self.index[k]} / {ing.id}")
                self.index[k] = ing.id
        self._keys = list(self.index)

    def __len__(self) -> int:
        return len(self.items)

    def lookup(self, name: str) -> tuple[Optional[Ingredient], str]:
        key = normalize(name)
        if not key:
            return None, "none"
        if key in self.index:
            return self.items[self.index[key]], "exact"
        # "Water (Aqua)" / "Cocos Nucifera (Coconut) Oil"：先去括号，再试括号内
        stripped = normalize(PAREN_RE.sub("", name))
        if stripped in self.index:
            return self.items[self.index[stripped]], "paren"
        for inner in PAREN_RE.findall(name):
            k = normalize(inner)
            if k in self.index:
                return self.items[self.index[k]], "paren"
        # 只对较长的名字做模糊匹配，避免短词误配
        if len(key) >= 5:
            hit = difflib.get_close_matches(key, self._keys, n=1, cutoff=self.fuzzy_cutoff)
            if hit:
                return self.items[self.index[hit[0]]], "fuzzy"
        return None, "none"

    def parse(self, text: str) -> list[ParsedItem]:
        parts = [p for p in SPLIT_RE.split(text) if normalize(p)]
        out = []
        for pos, raw in enumerate(parts):
            ing, how = self.lookup(raw)
            out.append(ParsedItem(raw=raw.strip(), position=pos, ingredient=ing, match=how))
        return out


def summarize_flags(items: list[ParsedItem]) -> list[tuple[str, str]]:
    """基于知识库的规则提示，返回 [(级别, 文案)]。级别：warn / note。"""
    flags = []
    known = [it for it in items if it.ingredient]

    frag = [it for it in known if {"fragrance", "fragrance_allergen"} & set(it.ingredient.tags)]
    if frag:
        names = "、".join(it.ingredient.zh for it in frag)
        flags.append(("warn", f"含香精/香料致敏原（{names}），敏感肌或眼周慎用。"))

    alc = [it for it in known if "drying_alcohol" in it.ingredient.tags and it.position < 5]
    if alc:
        flags.append(("warn", f"酒精排在第 {alc[0].position + 1} 位，含量可能较高，干皮/敏感肌可能感到刺激。"))

    for it in known:
        if it.ingredient.irritation >= 3:
            flags.append(("warn", f"{it.ingredient.zh}：{it.ingredient.note or '刺激性较高'}"))

    com = [it for it in known if it.ingredient.comedogenic >= 4]
    if com:
        names = "、".join(it.ingredient.zh for it in com)
        flags.append(("note", f"含传统致痘评分较高的成分（{names}），易长闭口/痘的话留意（评分仅供参考）。"))

    actives = {
        "retinoid": "维A类", "aha": "果酸(AHA)", "bha": "水杨酸(BHA)", "pha": "PHA",
        "vitamin_c_pure": "纯VC", "vitamin_c_derivative": "VC衍生物",
        "niacinamide": "烟酰胺", "copper_peptide": "铜肽", "benzoyl_peroxide": "过氧化苯甲酰",
    }
    found = []
    for tag, label in actives.items():
        if any(tag in it.ingredient.tags for it in known) and label not in found:
            found.append(label)
    if found:
        flags.append(("note", "主要活性成分：" + "、".join(found)))
    return flags

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

# 英文逗号：只有两侧都是数字时不切（1,2-己二醇），"Polyquaternium-7, Parfum" 要切开
COMMA_RE = r"(?<!\d),|,(?!\d)"
# 按逗号/顿号/分号/换行切分
SPLIT_RE = re.compile(COMMA_RE + r"|[，、;；\n]+")
# 国内标签「其他微量成分：」（含量 ≤ 0.1%）之后的成分标记为微量；TRACE_MARK 是 OCR 截取时统一写入的写法
TRACE_RE = re.compile(r"^\s*(?:其他|其它)?(?:微量|次要)成分\s*[:：]\s*")
TRACE_MARK = "其他微量成分："
TRACE_NOTE = "≤0.1%"
PAREN_RE = re.compile(r"[(（]([^)）]*)[)）]")
DIGITS_RE = re.compile(r"\d+")
COMPACT_RE = re.compile(r"[\s/.,\-]+")


def _balance_parens(s: str) -> str:
    """"Iron Oxides (CI 77491" 补上右括号，"CI 77499)" 去掉多余的右括号。"""
    s = s.strip()
    if s.count("(") > s.count(")"):
        return s + ")"
    if s.count(")") > s.count("(") and s.endswith(")"):
        return s[:-1]
    return s


def _typo_of(a: str, b: str) -> bool:
    """a 是否像 b 的拼写错误：数字相同、词数相同、每个词相同或（词头 3 个字符相同且相似度 ≥ 0.75）。"""
    if DIGITS_RE.findall(a) != DIGITS_RE.findall(b):
        return False
    wa, wb = a.split(), b.split()
    return len(wa) == len(wb) and all(
        x == y or (x[:3] == y[:3] and difflib.SequenceMatcher(None, x, y).ratio() >= 0.75) for x, y in zip(wa, wb))


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
    common_zh: str = ""  # 常用名（如「依克多因」），只用于显示；zh 是《已使用化妆品原料目录》标准名

    @property
    def display(self) -> str:
        """「常用名（标准名）」，没有常用名时只显示标准名。"""
        return f"{self.common_zh}（{self.zh}）" if self.common_zh else self.zh


@dataclass
class ParsedItem:
    raw: str
    position: int                     # 在成分表中的位置（0 起）；越靠前通常浓度越高
    ingredient: Optional[Ingredient]  # 未识别时为 None
    match: str                        # exact / paren / fuzzy / none
    trace: bool = False               # 在「其他微量成分：」之后（国内标签规定含量 ≤ 0.1%）


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
        # 标准名带拉丁名括号的，只写中文部分也能查到：「牛油果树（BUTYROSPERMUM PARKII）果脂」->「牛油果树果脂」
        for ing in self.items.values():
            if PAREN_RE.search(ing.zh):
                self.index.setdefault(normalize(PAREN_RE.sub("", ing.zh)), ing.id)
        self._keys = list(self.index)
        # 去掉空格和标点后的写法 -> id；两个成分撞车时不用（记为 None）
        self._compact: dict[str, Optional[str]] = {}
        for k, i in self.index.items():
            c = COMPACT_RE.sub("", k)
            self._compact[c] = i if self._compact.get(c, i) == i else None
        self._fuzzy: dict[str, Optional[str]] = {}  # 模糊匹配结果缓存（最慢的一步）

    def __len__(self) -> int:
        return len(self.items)

    def lookup(self, name: str) -> tuple[Optional[Ingredient], str]:
        key = normalize(name)
        if not key:
            return None, "none"
        if key in self.index:
            return self.items[self.index[key]], "exact"
        # "Benzylalcohol"、"Caprylic/ Capric Triglyceride"：只差空格和标点
        if self._compact.get(COMPACT_RE.sub("", key)):
            return self.items[self._compact[COMPACT_RE.sub("", key)]], "exact"
        # "Water (Aqua)" / "Cocos Nucifera (Coconut) Oil"：先去括号，再试括号内
        name = _balance_parens(name)
        stripped = normalize(PAREN_RE.sub("", name))
        if stripped in self.index:
            return self.items[self.index[stripped]], "paren"
        for inner in PAREN_RE.findall(name):
            k = normalize(inner)
            if k in self.index:
                return self.items[self.index[k]], "paren"
        # "Aqua / Water"、"CI 77891/Titanium Dioxide"：每一部分都精确命中同一个成分才算
        if "/" in key:
            ids = {self.index.get(normalize(p)) for p in key.split("/")}
            if len(ids) == 1 and None not in ids:
                return self.items[ids.pop()], "exact"
        # 只对较长的名字做模糊匹配，并且只容忍拼写错误：数字一致（laureth-7 ≠ laureth-4），
        # 逐词相近且词头相同（distearate ≠ stearate、acetate ≠ lactate）
        if len(key) >= 5:
            if key not in self._fuzzy:
                hits = difflib.get_close_matches(key, self._keys, n=5, cutoff=self.fuzzy_cutoff)
                self._fuzzy[key] = next((h for h in hits if _typo_of(key, h)), None)
            if self._fuzzy[key]:
                return self.items[self.index[self._fuzzy[key]]], "fuzzy"
        return None, "none"

    def parse(self, text: str) -> list[ParsedItem]:
        parts = [p for p in SPLIT_RE.split(text) if normalize(p)]
        out = []
        trace = False
        for raw in parts:
            m = TRACE_RE.match(raw)
            if m:
                trace, raw = True, raw[m.end():]
                if not normalize(raw):
                    continue
            ing, how = self.lookup(raw)
            pieces = [p for p in raw.split("/") if normalize(p)]
            if ing is None and len(pieces) > 1:
                # 「神经酰胺 NS/神经酰胺 NG」：整体查不到、但每一部分都认识，按多个成分处理
                hits = [self.lookup(p) for p in pieces]
                if all(i for i, _ in hits):
                    for p, (i, h) in zip(pieces, hits):
                        out.append(ParsedItem(raw=p.strip(), position=len(out), ingredient=i, match=h, trace=trace))
                    continue
            out.append(ParsedItem(raw=raw.strip(), position=len(out), ingredient=ing, match=how, trace=trace))
        return out


# note 里给维护者看的部分（出处、审核记录、待核实），不展示给用户
_INTERNAL_NOTE_RE = re.compile(r"待核实|目录|审核意见|Fulton|^刺激性 |^致痘性|^中文名")


def user_note(ing: Ingredient) -> str:
    """note 中面向用户的说明。"""
    return "；".join(p for p in ing.note.split("；") if p and not _INTERNAL_NOTE_RE.search(p))


def ingredient_hints(ing: Ingredient) -> list[str]:
    """单个成分的简短提示标签，供逐项展示。"""
    tags, hints = set(ing.tags), []
    if "fragrance_allergen" in tags:
        hints.append("香料致敏原")
    elif "fragrance" in tags:
        hints.append("香精/香料")
    if "drying_alcohol" in tags:
        hints.append("酒精")
    if "formaldehyde_releaser" in tags:
        hints.append("甲醛释放体")
    if ing.irritation >= 2:
        hints.append(f"刺激性{'高' if ing.irritation >= 3 else '中'}")
    if ing.comedogenic >= 3:
        hints.append(f"致痘评分 {ing.comedogenic}")
    return hints


def summarize_flags(items: list[ParsedItem]) -> list[tuple[str, str]]:
    """基于知识库的规则提示，返回 [(级别, 文案)]。级别：warn / note。"""
    flags = []
    known = [it for it in items if it.ingredient]

    frag = [it for it in known if {"fragrance", "fragrance_allergen"} & set(it.ingredient.tags)]
    if frag:
        names = "、".join(it.ingredient.display for it in frag)
        flags.append(("warn", f"含香精/香料致敏原（{names}），敏感肌或眼周慎用。"))

    alc = [it for it in known if "drying_alcohol" in it.ingredient.tags and it.position < 5]
    if alc:
        flags.append(("warn", f"酒精排在第 {alc[0].position + 1} 位，含量可能较高，干皮/敏感肌可能感到刺激。"))

    fr = [it for it in known if "formaldehyde_releaser" in it.ingredient.tags]
    if fr:
        names = "、".join(it.ingredient.display for it in fr)
        flags.append(("warn", f"含甲醛释放体防腐剂（{names}），敏感肌慎用。"))

    for it in known:
        if it.ingredient.irritation >= 3:
            note = user_note(it.ingredient)
            flags.append(("warn", f"{it.ingredient.display}：刺激/致敏性高" + (f"（{note}）" if note else "") + "。"))

    com = [it for it in known if it.ingredient.comedogenic >= 4]
    if com:
        names = "、".join(it.ingredient.display for it in com)
        flags.append(("note", f"含传统致痘评分较高的成分（{names}），易长闭口/痘的话留意（评分仅供参考）。"))

    actives = {
        "retinoid": "维A类", "aha": "果酸(AHA)", "bha": "水杨酸(BHA)", "pha": "PHA",
        "vitamin_c_pure": "纯VC", "vitamin_c_derivative": "VC衍生物",
        "niacinamide": "烟酰胺", "copper_peptide": "铜肽", "benzoyl_peroxide": "过氧化苯甲酰",
    }
    main, trace = [], []
    for tag, label in actives.items():
        hits = [it for it in known if tag in it.ingredient.tags]
        if not hits or label in main + trace:
            continue
        # 只出现在「其他微量成分」里的活性成分单独说明
        (trace if all(it.trace for it in hits) else main).append(label)
    if main or trace:
        parts = ["、".join(main)] if main else []
        if trace:
            parts.append("、".join(trace) + (f"（均为微量成分，{TRACE_NOTE}，作用可能有限）" if len(trace) > 1
                                              else f"（微量成分，{TRACE_NOTE}，作用可能有限）"))
        flags.append(("note", "含有的活性成分：" + "；".join(parts)))
    return flags


POPULATION_FOOTNOTE = "以上只是根据成分做的一般提示，不是针对个人或肤质的判断；有皮肤问题或特殊情况，建议咨询医生。"


def population_notes(items: list[ParsedItem]) -> list[tuple[str, str]]:
    """基于知识库规则的「需注意人群」提示，返回 [(人群, 说明)]。不做肤质预测，只给留意/慎用类提示。"""
    known = [it.ingredient for it in items if it.ingredient]
    names = lambda ings: "、".join(dict.fromkeys(i.display for i in ings))
    has = lambda *tags: [i for i in known if set(tags) & set(i.tags)]
    notes = []

    frag = has("fragrance", "fragrance_allergen")
    if frag:
        notes.append(("敏感肌", f"含香精/香料致敏原（{names(frag)}），慎用；初次使用可先在耳后或手臂内侧少量试用。"))
    fr = has("formaldehyde_releaser")
    if fr:
        notes.append(("敏感肌", f"含甲醛释放体防腐剂（{names(fr)}），慎用。"))
    alc = [it for it in items if it.ingredient and "drying_alcohol" in it.ingredient.tags and it.position < 5]
    if alc:
        notes.append(("干皮、敏感肌", f"酒精排在第 {alc[0].position + 1} 位，含量可能较高，留意使用后是否干燥、刺痛。"))
    com = [i for i in known if i.comedogenic >= 4]
    if com:
        notes.append(("易长闭口/痘痘者", f"含致痘评分较高的成分（{names(com)}），留意使用后是否长闭口（评分来自早期动物实验，仅供参考）。"))
    ret = [it for it in items if it.ingredient and "retinoid" in it.ingredient.tags]
    if ret and all(it.trace for it in ret):
        notes.append(("备孕、孕期、哺乳期",
                      f"含微量维A类（{TRACE_NOTE}），备孕、孕期、哺乳期建议咨询医生（{names([it.ingredient for it in ret])}）。"))
    elif ret:
        notes.append(("备孕、孕期、哺乳期",
                      f"含维A类（{names([it.ingredient for it in ret])}），一般建议避免使用，使用前请咨询医生。"))
    acid = {"aha", "bha", "pha"}
    strong = [it for it in items if it.ingredient and ((acid | {"vitamin_c_pure"}) & set(it.ingredient.tags)
                                                       or it.ingredient.irritation >= 3)]
    if strong:
        tags = {t for it in strong for t in it.ingredient.tags}
        label = "/".join(k for k, hit in [("酸类", acid & tags), ("纯VC", "vitamin_c_pure" in tags),
                                          ("高刺激成分", any(it.ingredient.irritation >= 3 for it in strong))] if hit)
        if all(it.trace for it in strong):
            # 措辞与微量维A类的孕期提示一致
            notes.append(("屏障受损、医美术后", f"含微量{label}（{TRACE_NOTE}），屏障受损、医美术后建议咨询医生"
                                             f"（{names([it.ingredient for it in strong])}）。"))
        else:
            marked = "、".join(dict.fromkeys(it.ingredient.display + ("（微量）" if it.trace else "") for it in strong))
            notes.append(("屏障受损、医美术后", f"含{label}（{marked}），慎用；术后护理请遵医嘱。"))
    return notes

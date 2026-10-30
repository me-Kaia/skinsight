"""护肤品叠用冲突检查（基于成分标签的规则引擎）。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .kb import TRACE_NOTE, ParsedItem

LEVEL_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
LEVEL_ZH = {"high": "高", "medium": "中", "low": "低", "info": "提示"}
DOWNGRADE = {"high": "medium", "medium": "low", "low": "info", "info": "info"}
EXFOLIANT = {"aha", "bha"}
TAG_ZH = {"retinoid": "维A类", "aha": "果酸", "bha": "水杨酸", "pha": "PHA", "vitamin_c_pure": "纯VC",
          "benzoyl_peroxide": "过氧化苯甲酰", "copper_peptide": "铜肽", "niacinamide": "烟酰胺"}


@dataclass(frozen=True)
class Rule:
    a: frozenset
    b: frozenset
    level: str
    message: str


def _r(a, b, level, msg):
    return Rule(frozenset(a), frozenset(b), level, msg)


RULES = [
    _r({"retinoid"}, EXFOLIANT, "high",
       "维A类 + 酸类：同一次叠用刺激明显叠加，容易泛红、脱皮。建议分早晚或隔天使用。"),
    _r({"retinoid"}, {"benzoyl_peroxide"}, "high",
       "维A类 + 过氧化苯甲酰：刺激叠加，且过氧化苯甲酰可能氧化部分维A类。建议分早晚使用。"),
    _r({"retinoid"}, {"retinoid"}, "medium",
       "两个产品都含维A类：相当于加量，刺激风险上升，一般选一个即可。"),
    _r({"vitamin_c_pure"}, {"benzoyl_peroxide"}, "medium",
       "纯VC + 过氧化苯甲酰：后者是氧化剂，会让VC失效。建议分早晚。"),
    _r({"vitamin_c_pure"}, EXFOLIANT, "medium",
       "纯VC + 酸类：两者都偏酸性，敏感肌叠用容易刺痛。建议分开时段。"),
    _r(EXFOLIANT, EXFOLIANT, "medium",
       "多个去角质产品叠用：容易过度去角质、损伤屏障。"),
    _r({"copper_peptide"}, {"vitamin_c_pure"} | EXFOLIANT, "low",
       "铜肽 + 纯VC/酸类：可能互相影响稳定性和效果（证据有限），保险起见分开时段。"),
    _r({"niacinamide"}, {"vitamin_c_pure"}, "info",
       "烟酰胺 + VC：「会互相抵消」是老说法，按正常护肤品的浓度和用法同用一般没问题；同用后泛红再分开。"),
]


@dataclass
class Finding:
    level: str
    products: tuple[str, ...]
    message: str


def product_tags(items: list[ParsedItem]) -> set[str]:
    return {t for it in items if it.ingredient for t in it.ingredient.tags}


def trace_only_tags(items: list[ParsedItem]) -> set[str]:
    """只由「其他微量成分」（≤0.1%）里的成分带来的标签。"""
    return product_tags([it for it in items if it.trace]) - product_tags([it for it in items if not it.trace])


def _match(rule: Rule, p1, p2) -> Finding | None:
    """p = (产品名, 标签, 只来自微量成分的标签)。规则按 p1→a、p2→b 的方向匹配；
    某一方触发规则的标签全部来自微量成分时，级别降一级，并在说明里注明。"""
    (n1, t1, r1), (n2, t2, r2) = p1, p2
    m1, m2 = rule.a & t1, rule.b & t2
    if not (m1 and m2):
        return None
    trace = [f"{n}中的{'、'.join(TAG_ZH.get(t, t) for t in sorted(m))}" for n, m, r in ((n1, m1, r1), (n2, m2, r2))
             if m <= r]
    if not trace:
        return Finding(rule.level, (n1, n2), rule.message)
    return Finding(DOWNGRADE[rule.level], (p1[0], p2[0]),
                   rule.message + f"（{'；'.join(trace)}只是微量成分，{TRACE_NOTE}，已降低一级）")


def check_routine(products: list[tuple[str, list[ParsedItem]]]) -> list[Finding]:
    """products: [(产品名, 解析后的成分表)]，检查两两之间的冲突。"""
    tags = [(name, product_tags(items), trace_only_tags(items)) for name, items in products]
    findings: list[Finding] = []

    for p1, p2 in combinations(tags, 2):
        for rule in RULES:
            f = _match(rule, p1, p2) or _match(rule, p2, p1)
            if f:
                findings.append(Finding(f.level, (p1[0], p2[0]), f.message))

    for name, t, _r in tags:
        if "retinoid" in t and EXFOLIANT & t:
            findings.append(Finding("info", (name,),
                                    "这个产品本身同时含维A类和酸类，按产品说明使用，不要再额外叠加其他酸类或维A。"))

    all_tags = set().union(*(t for _, t, _r in tags)) if tags else set()
    if all_tags & ({"retinoid", "vitamin_c_pure"} | EXFOLIANT):
        findings.append(Finding("info", tuple(n for n, _, _r in tags),
                                "使用维A类/酸类/纯VC期间皮肤对光更敏感，白天务必防晒。"))

    findings.sort(key=lambda f: LEVEL_ORDER[f.level])
    return findings

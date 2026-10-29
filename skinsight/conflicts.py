"""护肤品叠用冲突检查（基于成分标签的规则引擎）。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .kb import ParsedItem

LEVEL_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
LEVEL_ZH = {"high": "高", "medium": "中", "low": "低", "info": "提示"}
EXFOLIANT = {"aha", "bha"}


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


def check_routine(products: list[tuple[str, list[ParsedItem]]]) -> list[Finding]:
    """products: [(产品名, 解析后的成分表)]，检查两两之间的冲突。"""
    tags = [(name, product_tags(items)) for name, items in products]
    findings: list[Finding] = []

    for (n1, t1), (n2, t2) in combinations(tags, 2):
        for rule in RULES:
            if (rule.a & t1 and rule.b & t2) or (rule.a & t2 and rule.b & t1):
                findings.append(Finding(rule.level, (n1, n2), rule.message))

    for name, t in tags:
        if "retinoid" in t and EXFOLIANT & t:
            findings.append(Finding("info", (name,),
                                    "这个产品本身同时含维A类和酸类，按产品说明使用，不要再额外叠加其他酸类或维A。"))

    all_tags = set().union(*(t for _, t in tags)) if tags else set()
    if all_tags & ({"retinoid", "vitamin_c_pure"} | EXFOLIANT):
        findings.append(Finding("info", tuple(n for n, _ in tags),
                                "使用维A类/酸类/纯VC期间皮肤对光更敏感，白天务必防晒。"))

    findings.sort(key=lambda f: LEVEL_ORDER[f.level])
    return findings

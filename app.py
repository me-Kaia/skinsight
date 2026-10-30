"""skinsight 网页 demo（Gradio）：成分解读 + 叠用检查。

    python app.py            # 本地启动，默认 http://127.0.0.1:7860

只用知识库和规则，不加载肤质预测模型，也不需要 torch。
部署到魔搭创空间见 deploy/modelscope/。
"""
from __future__ import annotations

from skinsight.conflicts import LEVEL_ZH, check_routine
from skinsight.kb import KnowledgeBase, ingredient_hints, summarize_flags, user_note

KB = KnowledgeBase()

FUNC_ZH = {
    "abrasive": "摩擦剂", "absorbent": "吸附/控油", "anti-aging": "抗老", "antiacne": "祛痘", "antidandruff": "去屑",
    "antioxidant": "抗氧化", "antiperspirant": "止汗", "barrier": "修护屏障", "binder": "黏合剂", "brightening": "提亮",
    "chelating": "螯合剂", "colorant": "着色剂", "conditioning": "调理", "cooling": "清凉", "deodorant": "除臭",
    "emollient": "润肤", "emulsifier": "乳化剂", "exfoliant": "去角质", "film_former": "成膜剂", "fragrance": "香料",
    "hair_dye": "染发", "humectant": "保湿", "occlusive": "封闭保湿", "opacifier": "遮光/珠光", "oral_care": "口腔护理",
    "oxidizer": "氧化剂", "ph_adjuster": "pH 调节", "preservative": "防腐", "propellant": "推进剂", "solubilizer": "增溶剂",
    "solvent": "溶剂", "soothing": "舒缓", "stabilizer": "稳定剂", "sunscreen": "防晒", "surfactant": "表面活性剂/清洁",
    "sweetener": "甜味剂", "thickener": "增稠剂",
}
LEVEL_ICON = {"warn": "⚠️", "note": "ℹ️", "high": "🔴", "medium": "🟠", "low": "🟡", "info": "ℹ️"}
TABLE_HEADERS = ["#", "成分表原文", "名称", "作用", "提示"]

DISCLAIMER = (
    "**免责声明**：本工具仅供学习和信息参考，不构成医疗或皮肤科诊疗建议。成分作用和提示来自公开资料整理的知识库，"
    "可能有遗漏或错误；致痘评分来自早期动物实验，仅供参考；产品的实际表现取决于浓度和整体配方。"
    "皮肤出现持续不适请咨询医生。"
)

ANALYZE_EXAMPLES = [
    "水、甘油、丁二醇、烟酰胺、透明质酸钠、1,2-己二醇、香精、芳樟醇、苯氧乙醇",
    "Water, Alcohol Denat., Ethylhexyl Methoxycinnamate, Octocrylene, Glycerin, Ethylhexyl Palmitate, "
    "Dimethicone, Parfum, Hexyl Cinnamal, Limonene, DMDM Hydantoin",
    "Aqua, Squalane, Caprylic/Capric Triglyceride, Retinol, Bakuchiol, Tocopherol, Ceramide NP, Ectoin, "
    "Phenoxyethanol, Ethylhexylglycerin",
]
CHECK_EXAMPLES = [
    ["A醇精华", "水, 视黄醇, 角鲨烷, 生育酚", "水杨酸水", "水, 水杨酸, 烟酰胺, 丁二醇", "", "", "", ""],
    ["VC 精华", "Water, Ascorbic Acid, Ferulic Acid, Tocopherol", "烟酰胺精华", "Water, Niacinamide, Zinc PCA, Glycerin",
     "防晒霜", "Water, Zinc Oxide, Titanium Dioxide, Dimethicone", "", ""],
    ["祛痘凝胶", "Water, Benzoyl Peroxide, Carbomer", "视黄醛面霜", "Water, Retinal, Squalane, Glycerin",
     "果酸爽肤水", "Water, Glycolic Acid, Glycerin", "", ""],
]


def analyze(text: str) -> tuple[str, list[list]]:
    """成分表 -> (Markdown 摘要, 逐成分表格行)。"""
    items = KB.parse(text or "")
    if not items:
        return "请先粘贴一个成分表（中文或英文 INCI，用逗号、顿号或换行分隔）。", []
    known = sum(1 for it in items if it.ingredient)
    lines = [f"共 **{len(items)}** 个成分，知识库识别 **{known}** 个。"]
    flags = summarize_flags(items)
    if flags:
        lines += ["", "**提示**"] + [f"- {LEVEL_ICON[level]} {msg}" for level, msg in flags]
    else:
        lines += ["", "没有发现需要特别提示的成分。"]
    rows = []
    for it in items:
        ing = it.ingredient
        if ing is None:
            rows.append([it.position + 1, it.raw, "（未收录）", "", ""])
            continue
        name = ing.display + ("（模糊匹配）" if it.match == "fuzzy" else "")
        func = "、".join(FUNC_ZH.get(f, f) for f in ing.functions)
        hints = "、".join(ingredient_hints(ing))
        note = user_note(ing)
        rows.append([it.position + 1, it.raw, name, func, "；".join(x for x in (hints, note) if x)])
    return "\n".join(lines), rows


def check(*fields: str) -> str:
    """fields：产品名、成分表交替排列（最多 4 个产品）。成分表为空的产品跳过。"""
    products = []
    for i in range(0, len(fields), 2):
        name, text = (fields[i] or "").strip(), (fields[i + 1] or "").strip()
        if text:
            products.append((name or f"产品{i // 2 + 1}", KB.parse(text)))
    if len(products) < 2:
        return "请至少填写 2 个产品的成分表。"
    names = [n for n, _ in products]
    if len(set(names)) < len(names):
        return "产品名有重复，请给每个产品起不同的名字。"
    findings = check_routine(products)
    if not findings:
        return f"检查了 {len(products)} 个产品，没有发现已知的叠用冲突。"
    lines = [f"检查了 {len(products)} 个产品，发现 {len(findings)} 条提示：", ""]
    for f in findings:
        lines.append(f"- {LEVEL_ICON[f.level]} **[{LEVEL_ZH[f.level]}] {' + '.join(f.products)}**：{f.message}")
    return "\n".join(lines)


def build_demo():
    import gradio as gr

    with gr.Blocks(title="skinsight 成分解读") as demo:
        gr.Markdown("# skinsight 护肤品成分解读\n粘贴成分表（中文或英文 INCI 都可以），查看每个成分的作用和需要留意的地方，"
                    "或者检查几个产品一起用会不会冲突。")
        with gr.Tab("成分解读"):
            text = gr.Textbox(label="成分表", lines=5, placeholder="例如：水、甘油、烟酰胺、香精……")
            btn = gr.Button("解读", variant="primary")
            summary = gr.Markdown()
            table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["number", "str", "str", "str", "str"],
                                 interactive=False, wrap=True)
            gr.Examples(ANALYZE_EXAMPLES, inputs=text, label="示例")
            btn.click(analyze, inputs=text, outputs=[summary, table])
            text.submit(analyze, inputs=text, outputs=[summary, table])
        with gr.Tab("叠用检查"):
            gr.Markdown("填写 2–4 个准备一起用的产品（同一天早晚叠用或同一次叠涂），成分表留空的产品不参与检查。")
            fields = []
            for i in range(4):
                with gr.Row():
                    fields.append(gr.Textbox(label=f"产品 {i + 1} 名称", scale=1, max_lines=1, placeholder=f"产品{i + 1}"))
                    fields.append(gr.Textbox(label=f"产品 {i + 1} 成分表", scale=4, lines=2))
            cbtn = gr.Button("检查", variant="primary")
            result = gr.Markdown()
            gr.Examples(CHECK_EXAMPLES, inputs=fields, label="示例")
            cbtn.click(check, inputs=fields, outputs=result)
        gr.Markdown(DISCLAIMER)
    return demo


if __name__ == "__main__":
    build_demo().launch()

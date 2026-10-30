"""skinsight 网页 demo（Gradio）：成分解读（可拍照识别）+ 叠用检查。

    python app.py            # 本地启动，默认 http://127.0.0.1:7860

只用知识库和规则，不加载肤质预测模型，也不需要 torch。拍照识别用本地 OCR（RapidOCR），
上传的图片只在内存里处理（可先裁剪），识别前删除 Gradio 写到缓存目录的上传文件。
部署到魔搭创空间见 deploy/modelscope/。
"""
from __future__ import annotations

import os
import re

from skinsight.conflicts import LEVEL_ZH, check_routine
from skinsight.iecic import Catalog
from skinsight.kb import (POPULATION_FOOTNOTE, TRACE_NOTE, KnowledgeBase, ingredient_hints, population_notes,
                          summarize_flags, user_note)

KB = KnowledgeBase()
CATALOG = Catalog()  # 《已使用化妆品原料目录》备用字典：知识库没收录的成分用它确认、给疑似错字找候选
ARROW = " → "

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


PHOTO_TIPS = ("**拍照提示**：把包装放平或捏平，对准焦距，避开灯光反光，尽量只拍成分表那一块。"
              "截图里有瓶身、其他文字时，上传后点编辑器左侧的图片按钮 → 「Crop」裁剪，拖动边框只框住成分表，"
              "点「Confirm crop」确认后再识别。"
              "瓶身是圆的时，让文字行保持水平、正对中间拍；成分表较长可以转动瓶子分段拍，勾选「追加到已有文本」逐段识别。"
              "识别结果会填进下面的文本框，请核对、改正后再点「分析」。")


def _load_image(value):
    """图片组件的值 -> numpy 图像（BGR）。

    value 可以是文件路径（读入内存后立即删除文件），也可以是 ImageEditor 的 {"composite": ndarray, ...}
    （裁剪后的结果，RGB 数组，本来就在内存里）。
    """
    import numpy as np
    from skinsight.ocr import image_from_bytes
    if isinstance(value, dict):
        value = value.get("composite") if value.get("composite") is not None else value.get("background")
    if value is None:
        return None
    if isinstance(value, str):
        try:
            with open(value, "rb") as f:
                data = f.read()
        finally:
            try:
                os.remove(value)  # 不保存用户图片
            except OSError:
                pass
        return image_from_bytes(data)
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA：透明背景按白色处理
        alpha = arr[:, :, 3:4] / 255.0
        arr = (arr[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, ::-1]) if arr.ndim == 3 else arr


def recognize_photo(value, current: str = "", append: bool = False) -> tuple[str, str]:
    """上传（并可能裁剪过）的照片 -> (成分表文本, 状态说明)。图片只在内存里处理。

    append=True 时把识别结果接在 current 后面（瓶身弯曲时分段拍照用）。
    """
    try:
        from skinsight.ocr import extract_ingredients, recognize
        img = _load_image(value)
        if img is None:
            return current, "请先上传一张成分表照片。"
        lines = recognize(img)
    except ImportError:
        return current, "当前环境没有安装 OCR（rapidocr_onnxruntime），请直接粘贴成分表文字。"
    except ValueError as e:
        return current, str(e)
    if not lines:
        return current, "没有识别到文字，请换一张更清晰、正对成分表的照片。"
    text, found = extract_ingredients(lines, KB, CATALOG)
    if found:
        status = f"识别到 {len(lines)} 行文字，已截取「成分」后面的部分。OCR 可能认错字，请核对后点「分析」。"
    else:
        status = (f"识别到 {len(lines)} 行文字，但没找到「成分 / 全成分 / Ingredients」标题，已填入全部文字，"
                  "请删掉成分表以外的内容后再点「分析」。")
    if append and current.strip():
        sep = "、" if re.search(r"[一-鿿]", current[-1:] + text[:1]) else ", "
        text = current.rstrip(" ,，、") + sep + text
    unknown = [it.raw for it in KB.parse(text) if not it.ingredient and not CATALOG.lookup(it.raw)]
    if unknown:
        status += f"其中 {len(unknown)} 个成分知识库和原料目录都对不上，可能是 OCR 认错了字，点「分析」后可以查看候选名称。"
    return text, status


def classify(it):
    """解析结果 -> (类别, 目录条目)。类别：known 知识库已解读 / catalog 目录登记原料 / unknown 未识别。"""
    if it.ingredient:
        return "known", None
    entry = CATALOG.lookup(it.raw)
    return ("catalog", entry) if entry else ("unknown", None)


def suggestions_for(items) -> list[str]:
    """未识别成分的候选替换，「原文 → 建议」。只给候选，由用户勾选确认。"""
    out = []
    for it in items:
        if classify(it)[0] == "unknown":
            out += [f"{it.raw}{ARROW}{s}" for s in CATALOG.suggest(it.raw, kb=KB)]
    return list(dict.fromkeys(out))


def apply_suggestions(text: str, chosen: list[str]) -> str:
    """把用户勾选的「原文 → 建议」替换进成分表文本（每个原文只替换一次）。"""
    done = set()
    for c in chosen or []:
        raw, _, new = c.partition(ARROW)
        if raw and new and raw not in done and raw in text:
            text = text.replace(raw, new, 1)
            done.add(raw)
    return text


def population_md(items) -> str:
    """「需注意人群」板块。"""
    notes = population_notes(items)
    lines = ["**需注意人群**", ""]
    lines += [f"- **{who}**：{msg}" for who, msg in notes] or ["没有触发需要特别留意的规则。"]
    return "\n".join(lines + ["", POPULATION_FOOTNOTE])


def highlight(items) -> list[tuple[str, str | None]]:
    """供 HighlightedText 显示：知识库已解读的不标色，目录登记原料标黄，未识别的标红。"""
    labels = {"known": None, "catalog": "登记原料，暂无解读", "unknown": "未识别"}
    out = []
    for it in items:
        if out:
            out.append(("、", None))
        out.append((it.raw, labels[classify(it)[0]]))
    return out


def analyze(text: str) -> tuple[str, str, list, list[list], list[str]]:
    """成分表 -> (Markdown 摘要, 需注意人群, 高亮, 逐成分表格行, 疑似错字的候选替换)。"""
    items = KB.parse(text or "")
    if not items:
        return "请先粘贴一个成分表（中文或英文 INCI，用逗号、顿号或换行分隔），或上传成分表照片。", "", [], [], []
    kinds = [classify(it) for it in items]
    n_known = sum(k == "known" for k, _ in kinds)
    n_cat = sum(k == "catalog" for k, _ in kinds)
    n_unk = len(items) - n_known - n_cat
    lines = [f"共 **{len(items)}** 个成分：知识库已解读 **{n_known}** 个，"
             f"原料目录登记但暂无解读 **{n_cat}** 个，未识别 **{n_unk}** 个。"]
    n_trace = sum(it.trace for it in items)
    if n_trace:
        lines.append(f"其中 **{n_trace}** 个在「其他微量成分」之后，按国内标签规定含量 {TRACE_NOTE}，表格里标为「微量」。")
    if n_unk:
        lines.append("未识别的成分可能是 OCR 认错了字，请对照包装核对；下方有原料目录里的候选名称可以勾选替换。")
    flags = summarize_flags(items)
    if flags:
        lines += ["", "**提示**"] + [f"- {LEVEL_ICON[level]} {msg}" for level, msg in flags]
    else:
        lines += ["", "没有发现需要特别提示的成分。"]
    rows = []
    for it, (kind, entry) in zip(items, kinds):
        ing = it.ingredient
        if kind == "catalog":
            rows.append([it.position + 1, it.raw, entry.zh, "", "已确认是登记原料，暂无详细解读"])
            continue
        if kind == "unknown":
            cands = CATALOG.suggest(it.raw, kb=KB)
            rows.append([it.position + 1, it.raw, "⚠️ 未识别", "",
                         ("疑似：" + " / ".join(cands)) if cands else "知识库和原料目录里都没有，请核对"])
            continue
        name = ing.display + ("（模糊匹配）" if it.match == "fuzzy" else "")
        func = "、".join(FUNC_ZH.get(f, f) for f in ing.functions)
        hints = "、".join(ingredient_hints(ing))
        note = user_note(ing)
        rows.append([it.position + 1, it.raw, name, func, "；".join(x for x in (hints, note) if x)])
    for it, row in zip(items, rows):
        if it.trace:
            row[4] = "；".join(x for x in (f"微量（{TRACE_NOTE}）", row[4]) if x)
    return "\n".join(lines), population_md(items), highlight(items), rows, suggestions_for(items)


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


def _delete_uploads_after_preprocess(editor):
    """ImageEditor 取到图像数组后，立即删除 Gradio 写到缓存目录的上传文件。

    不用子类：子类会被前端当成自定义组件去加载、导致页面打不开，所以只在实例上包一层 preprocess。
    """
    original = editor.preprocess

    def preprocess(payload):
        try:
            return original(payload)
        finally:
            files = [payload.background, payload.composite, *(payload.layers or [])] if payload else []
            for fd in files:
                path = getattr(fd, "path", None)
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
    editor.preprocess = preprocess
    return editor


def build_demo():
    import gradio as gr

    # delete_cache：万一有上传文件没被删掉，每 2 分钟清理一次超过 2 分钟的缓存
    with gr.Blocks(title="skinsight 成分解读", delete_cache=(120, 120)) as demo:
        gr.Markdown("# skinsight 护肤品成分解读\n粘贴成分表（中文或英文 INCI 都可以），查看每个成分的作用和需要留意的地方，"
                    "或者检查几个产品一起用会不会冲突。")
        with gr.Tab("成分解读"):
            with gr.Accordion("拍照识别成分表（可选）", open=True):
                gr.Markdown(PHOTO_TIPS)
                photo = _delete_uploads_after_preprocess(gr.ImageEditor(
                    label="成分表照片（上传后可裁剪，只保留成分表区域）", type="numpy", sources=["upload", "webcam"],
                    transforms=("crop",), brush=False, eraser=False, layers=False, height=420))
                with gr.Row():
                    ocr_btn = gr.Button("识别图片文字")
                    append = gr.Checkbox(label="追加到已有文本（分段拍照时用）", value=False)
                ocr_status = gr.Markdown()
            text = gr.Textbox(label="成分表（可编辑）", lines=5, placeholder="例如：水、甘油、烟酰胺、香精……")
            btn = gr.Button("分析", variant="primary")
            summary = gr.Markdown()
            marked = gr.HighlightedText(label="成分识别情况（红色：未识别，黄色：原料目录登记但暂无解读）",
                                        color_map={"未识别": "red", "登记原料，暂无解读": "yellow"},
                                        show_legend=False, combine_adjacent=False)
            with gr.Group(visible=False) as fix_box:
                fixes = gr.CheckboxGroup(label="疑似 OCR 错字：勾选正确的名称后点「替换并重新分析」（不会自动替换）",
                                         choices=[])
                fix_btn = gr.Button("替换并重新分析")
            people = gr.Markdown()
            table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["number", "str", "str", "str", "str"],
                                 interactive=False, wrap=True)
            gr.Examples(ANALYZE_EXAMPLES, inputs=text, label="示例")

            def run(t):
                s, p, m, rows, sugg = analyze(t)
                return s, p, m, rows, gr.update(choices=sugg, value=[]), gr.update(visible=bool(sugg))

            outs = [summary, people, marked, table, fixes, fix_box]
            # 识别后清空图片组件，图片不再留在页面上
            ocr_btn.click(recognize_photo, inputs=[photo, text, append], outputs=[text, ocr_status]).then(
                lambda: None, outputs=photo)
            btn.click(run, inputs=text, outputs=outs)
            text.submit(run, inputs=text, outputs=outs)
            fix_btn.click(apply_suggestions, inputs=[text, fixes], outputs=text).then(run, inputs=text, outputs=outs)
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

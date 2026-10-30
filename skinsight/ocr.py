"""拍照识别成分表：RapidOCR 识别文字 -> 还原阅读顺序 -> 截取成分段落 -> 拼接断行的成分名。

OCR 用 rapidocr_onnxruntime（本地运行，模型随 pip 包一起安装，不需要 API key）。
文字处理部分（lines_from_ocr / extract_ingredients）不依赖 OCR，可以单独测试。
"""
from __future__ import annotations

import math
import re

from .kb import COMMA_RE, normalize

# 成分段开头：「全成分：」「成分：」「配料」「Ingredients:」「INCI:」……（「其他微量成分：」不算）
HEADER_RE = re.compile(
    r"(全成分(?:表)?|成分表|产品成分|(?<!微量)(?<!其他)(?<!其它)(?<!次要)成分|配料|ingredients?|inci)\s*[:：]?", re.I)
# 成分表中间的小标题，当作分隔符
SUBHEAD_RE = re.compile(r"(?:其他|其它)?(?:微量|次要)成分\s*[:：]?|(?:其他|其它)成分\s*[:：]")
# 成分段之后常见的其他栏目，遇到就截断
STOP_RE = re.compile(
    r"(使用方法|用法|使用说明|注意事项|警示|警告|保质期|限期使用日期|生产日期|生产批号|批号|净含量|规格|执行标准|"
    r"生产企业|生产商|委托方|受托方|经销商|产地|原产国|地址|电话|网址|备案|许可证|贮存|储存|功效|"
    r"directions|how to use|to use|warnings?|cautions?|made in|manufactured|distributed|net wt|net weight|"
    r"best before|exp\b|lot\b|batch|keep out|for external use|store\b|storage)\s*[:：]?",
    re.I)
CJK_RE = re.compile(r"[一-鿿]")
SEP_END = ("，", ",", "、", ";", "；", "/", "(", "（")
SEP_RE = re.compile(COMMA_RE + r"|[，、;；]")  # 与 kb.SPLIT_RE 一致：1,2-己二醇 的逗号不算分隔符
_compact = lambda s: re.sub(r"[\s/.,\-]+", "", normalize(s))


def _is_header(text: str) -> bool:
    m = HEADER_RE.search(text)
    return bool(m) and (m.group(0).rstrip().endswith((":", "：")) or m.start() == 0)


def _rows(items):
    """items: [(y, x_left, x_right, h, text)] -> 按竖直位置聚类成行 [[y, h, [item, ...]], ...]。"""
    rows: list[list] = []
    for it in sorted(items):
        y, h = it[0], it[3]
        if rows and abs(y - rows[-1][0]) < 0.5 * max(h, rows[-1][1]):
            rows[-1][2].append(it)
        else:
            rows.append([y, h, [it]])
    return rows


def _same_column(items):
    """只保留和「成分」标题在同一列的文字框（横向重叠至少一半），丢掉左右两侧的瓶身、图片里的字。

    标题和后面的文字可能分在两个框里（「成分:」单独一格），把同一行紧挨着的框一起算进列宽。
    找不到标题时原样返回。
    """
    for _y, h, parts in _rows(items):
        parts = sorted(parts, key=lambda it: it[1])
        for k, (_, left, right, _, text) in enumerate(parts):
            if not _is_header(text):
                continue
            for _, ql, qr, _, _ in parts[k + 1:]:
                if ql - right > 2 * h:
                    break
                right = qr
            lo, hi = left - 0.5 * h, right + 0.5 * h
            return [it for it in items if min(it[2], hi) - max(it[1], lo) >= 0.5 * (it[2] - it[1])]
    return items


def lines_from_ocr(result, column: bool = True) -> list[str]:
    """RapidOCR 的 [[box, text, score], ...] -> 按阅读顺序排好的文本行。

    照片常常是斜的：先用长文本框的上边沿估计整体倾角，把框旋转回水平，
    再按竖直位置聚类成行（同一行可能被切成多个框），行内按横坐标排序。
    column=True 时先做版面分栏，只保留「成分」标题所在的那一列（见 _same_column）。
    """
    boxes = [(box, text.strip()) for box, text, _score in result or [] if text.strip()]
    if not boxes:
        return []
    angles = []
    for (tl, tr, _br, bl), _text in boxes:
        width, height = math.dist(tl, tr), math.dist(tl, bl)
        if width > 3 * height:
            angles.append(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
    skew = sorted(angles)[len(angles) // 2] if angles else 0.0
    cos, sin = math.cos(skew), math.sin(skew)
    items = []
    for box, text in boxes:
        xs = [px * cos + py * sin for px, py in box]
        ys = [-px * sin + py * cos for px, py in box]
        items.append((sum(ys) / 4, min(xs), max(xs), math.dist(box[0], box[3]), text))
    if column:
        items = _same_column(items)
    out = []
    for _y, _h, parts in _rows(items):
        line = ""
        for it in sorted(parts, key=lambda it: it[1]):
            line = _join(line, it[4], same_line=True)
        out.append(line)
    return out


def _join(a: str, b: str, same_line: bool = False, kb=None) -> str:
    """拼接两段文字：中文之间不加空格；英文断行时补空格；连字符断词接回去。

    换行处没有分隔符时，可能是一个成分名被折行，也可能是漏识别了逗号。给了 kb 时用知识库判断：
    拼起来是已知成分就拼接；拼起来不认识、但两半各自是已知成分，就补一个分隔符。
    """
    if not a:
        return b
    if not b:
        return a
    if a.endswith(SEP_END) or b.startswith((")", "）", "，", ",", "、", ";", "；")):
        return a + b
    if a.endswith("-") and not same_line:
        # 字母之间的连字符是排版断词（"Trigly-" + "ceride"、"（BUTYROSPE-" + "RMUM PARKII）"），去掉；
        # "PEG-" + "100 Stearate" 保留
        return a[:-1] + b if len(a) > 1 and a[-2].isalpha() and b[0].isalpha() else a + b
    if not same_line and re.search(r"\d$", a) and re.match(r"\d+-", b):
        return a + "," + b  # "1" + "2-己二醇"：行尾的逗号没识别出来
    cjk = bool(CJK_RE.match(a[-1]) or CJK_RE.match(b[0]))
    glue = "" if cjk else " "
    if kb is not None and not same_line:
        tail, head = SEP_RE.split(a)[-1].strip(), SEP_RE.split(b)[0].strip()
        if tail and head:
            if not cjk and " " not in tail and kb.lookup(tail + head)[0]:
                glue = ""  # OCR 丢了空格时："SodiumHydro" + "xide"、"积雪草（CE" + "NTELLA…"
            elif not kb.lookup(tail + glue + head)[0] and kb.lookup(tail)[0] and kb.lookup(head)[0]:
                return a + ("、" if cjk else ", ") + b
    return a + glue + b


def _restore_spaces(text: str, kb, catalog=None) -> str:
    """OCR 常把成分名里的空格丢掉：英文名（"ButyleneGlycol"）、中文名括号里的拉丁名（"牛油果树（BUTYROSPERMUMPARKII）果脂"）。

    只差空格/标点就能对上知识库或原料目录的，换回标准写法；有别的字不一样的不改（那是疑似错字，交给用户确认）。
    """
    same = lambda a, b: _compact(a) == _compact(b)

    def fix(seg: str) -> str:
        s = seg.strip()
        if not s:
            return seg
        if CJK_RE.search(s):
            if not re.search(r"[(（][^)）]*[A-Za-z]{4}", s):
                return seg
            ing, how = kb.lookup(s)
            if ing and how in ("exact", "paren") and same(s, ing.zh):
                return seg.replace(s, ing.zh)
            entry = catalog.lookup(s) if catalog is not None else None
            return seg.replace(s, entry.zh) if entry and same(s, entry.zh) else seg
        if " " in s or len(s) < 8:
            return seg
        ing, how = kb.lookup(s)
        if ing and how == "exact" and same(s, ing.inci):
            return seg.replace(s, ing.inci)
        return seg
    parts = re.split(f"({SEP_RE.pattern})", text)
    return "".join(p if SEP_RE.fullmatch(p or "") else fix(p) for p in parts)


def extract_ingredients(lines: list[str], kb=None, catalog=None) -> tuple[str, bool]:
    """从 OCR 文本行里截出成分表。返回 (成分表文本, 是否找到了「成分」标题)。

    找到标题时取标题之后、下一个栏目（使用方法、保质期、Directions……）之前的内容；
    没找到时返回全部文字，交给用户在文本框里删改。断开的成分名按 _join 的规则拼回去，
    「其他微量成分：」这类小标题换成分隔符。
    """
    text_lines = [l.strip() for l in lines if l.strip()]
    start, found = 0, False
    for i, line in enumerate(text_lines):
        m = HEADER_RE.search(line)
        if m and (m.group(0).rstrip().endswith((":", "：")) or m.start() == 0):
            start, found = i, True
            text_lines[i] = line[m.end():]
            break
    body = []
    for line in text_lines[start:]:
        m = STOP_RE.search(line)
        if found and m and (m.start() == 0 or line[:m.start()].rstrip().endswith(("。", "."))):
            if m.start() > 0:
                body.append(line[:m.start()])
            break
        body.append(line)
    text = ""
    for line in body:
        text = _join(text, SUBHEAD_RE.sub("、", line).strip(), kb=kb)
    text = re.sub(r"[、，,]\s*[、，,]", "、", text)  # 小标题前后本来就有分隔符时去重
    text = re.sub(r"(?<=[一-鿿）)])\s*[.。]\s*(?=\S)", "、", text)  # 中文后面的句点多半是认错的逗号
    text = re.sub(r"\s*([，、;；])\s*", r"\1", text)          # 中文分隔符前后不留空格
    text = re.sub(r"\s*,\s*(?!\d)|(?<!\d)\s*,\s*", ", ", text)  # 英文逗号后统一一个空格（1,2-己二醇 不动）
    text = re.sub(r"\s{2,}", " ", text).strip(" .。;；,，")
    if kb is not None:
        text = _restore_spaces(text, kb, catalog)
    return text, found


_ENGINE = None


def recognize(image) -> list[str]:
    """image：numpy 数组（BGR，cv2.imdecode 的结果）。返回按阅读顺序的文本行。首次调用时加载模型。"""
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    result, _elapse = _ENGINE(image)
    return lines_from_ocr(result)


def image_from_bytes(data: bytes):
    """图片字节 -> numpy 数组，只在内存里解码。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取图片，请上传 JPG 或 PNG 格式的照片")
    return img

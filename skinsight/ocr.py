"""拍照识别成分表：RapidOCR 识别文字 -> 还原阅读顺序 -> 截取成分段落 -> 拼接断行的成分名。

OCR 用 rapidocr_onnxruntime（本地运行，模型随 pip 包一起安装，不需要 API key）。
文字处理部分（lines_from_ocr / extract_ingredients）不依赖 OCR，可以单独测试。
"""
from __future__ import annotations

import math
import re

from .kb import COMMA_RE

# 成分段开头：「全成分：」「成分：」「配料」「Ingredients:」「INCI:」……
HEADER_RE = re.compile(r"(全成分(?:表)?|成分表|产品成分|成分|配料|ingredients?|inci)\s*[:：]?", re.I)
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


def lines_from_ocr(result) -> list[str]:
    """RapidOCR 的 [[box, text, score], ...] -> 按阅读顺序排好的文本行。

    照片常常是斜的：先用长文本框的上边沿估计整体倾角，把框中心旋转回水平，
    再按竖直位置聚类成行（同一行可能被切成多个框），行内按横坐标排序。
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
    for (tl, tr, br, bl), text in boxes:
        cx, cy = (tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2
        items.append((-cx * sin + cy * cos, cx * cos + cy * sin, math.dist(tl, bl), text))
    items.sort()
    rows: list[list] = []
    for y, x, h, text in items:
        if rows and abs(y - rows[-1][0]) < 0.5 * max(h, rows[-1][1]):
            rows[-1][2].append((x, text))
        else:
            rows.append([y, h, [(x, text)]])
    out = []
    for _y, _h, parts in rows:
        parts.sort()
        line = ""
        for _x, text in parts:
            line = _join(line, text, same_line=True)
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
        # "Trigly-" + "ceride" 是断词，去掉连字符；"PEG-" + "100 Stearate" 保留
        return a[:-1] + b if b[0].islower() else a + b
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


def _restore_spaces(text: str, kb) -> str:
    """OCR 常把英文成分名的空格丢掉（"ButyleneGlycol"）：只差空格/标点就能对上知识库的，换回标准 INCI 写法。"""
    def fix(seg: str) -> str:
        s = seg.strip()
        if not s or " " in s or CJK_RE.search(s) or len(s) < 8:
            return seg
        ing, how = kb.lookup(s)
        if ing and how == "exact" and re.sub(r"[\s/.,\-]+", "", ing.inci.lower()) == re.sub(r"[\s/.,\-]+", "", s.lower()):
            return seg.replace(s, ing.inci)
        return seg
    parts = re.split(f"({COMMA_RE})", text)
    return "".join(p if re.fullmatch(COMMA_RE, p or "") else fix(p) for p in parts)


def extract_ingredients(lines: list[str], kb=None) -> tuple[str, bool]:
    """从 OCR 文本行里截出成分表。返回 (成分表文本, 是否找到了「成分」标题)。

    找到标题时取标题之后、下一个栏目（使用方法、保质期、Directions……）之前的内容；
    没找到时返回全部文字，交给用户在文本框里删改。断开的成分名按 _join 的规则拼回去。
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
        text = _join(text, line.strip(), kb=kb)
    text = re.sub(r"\s*([，、;；])\s*", r"\1", text)          # 中文分隔符前后不留空格
    text = re.sub(r"\s*,\s*(?!\d)|(?<!\d)\s*,\s*", ", ", text)  # 英文逗号后统一一个空格（1,2-己二醇 不动）
    text = re.sub(r"\s{2,}", " ", text).strip(" .。;；,，")
    if kb is not None:
        text = _restore_spaces(text, kb)
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

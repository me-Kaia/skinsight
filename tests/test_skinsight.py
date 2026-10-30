import importlib.util
import math
import unittest

from skinsight.conflicts import check_routine
import csv
import os
import tempfile

from skinsight.data_utils import (CATEGORIES, LABELS, Vocab, auc_scores, clean_ingredients,
                                  kb_feature_matrix, kb_feature_names, load_cosmetics_csv,
                                  make_synthetic, multiclass_scores, split_samples, to_tokens)
from skinsight.kb import KnowledgeBase

try:
    import torch
except ImportError:
    torch = None

kb = KnowledgeBase()


class TestKB(unittest.TestCase):
    def test_split_keeps_numeric_commas(self):
        items = kb.parse("水, 1,2-己二醇, 甘油")
        self.assertEqual([i.ingredient.id for i in items], ["water", "hexanediol", "glycerin"])

    def test_split_comma_rule(self):
        raw = lambda t: [i.raw for i in kb.parse(t)]
        self.assertEqual(raw("1,2-hexanediol"), ["1,2-hexanediol"])  # 两侧都是数字：不切
        self.assertEqual(raw("polyquaternium-7, parfum"), ["polyquaternium-7", "parfum"])
        self.assertEqual(raw("Steareth-10,Dimethicone"), ["Steareth-10", "Dimethicone"])
        self.assertEqual(raw("Water,1,2-Hexanediol"), ["Water", "1,2-Hexanediol"])
        self.assertEqual(clean_ingredients("Polyquaternium-7, Parfum, 1,2-Hexanediol"),
                         "Polyquaternium-7, Parfum, 1,2-Hexanediol")

    def test_chinese_and_inci_map_to_same_id(self):
        self.assertEqual(to_tokens("烟酰胺，透明质酸钠", kb), to_tokens("Niacinamide, Sodium Hyaluronate", kb))

    def test_parenthesis_and_fuzzy(self):
        self.assertEqual(kb.lookup("Water (Aqua)")[0].id, "water")
        ing, how = kb.lookup("Niacinamid")
        self.assertEqual((ing.id, how), ("niacinamide", "fuzzy"))
        self.assertIsNone(kb.lookup("Totally Unknown Thing")[0])

    def test_lookup_variants(self):
        ids = lambda *names: [getattr(kb.lookup(n)[0], "id", None) for n in names]
        # 只差空格/标点、斜杠并列写法、括号不闭合
        self.assertEqual(ids("Benzylalcohol", "Caprylic/ Capric Triglyceride", "CI 77891/Titanium Dioxide",
                             "Iron Oxides (CI 77491", "CI 77499)"),
                         ["benzyl_alcohol", "caprylic_capric_tg", "titanium_dioxide", "ci_77491", "ci_77499"])
        self.assertIsNone(kb.lookup("Dimethicone/Vinyl Dimethicone Crosspolymer")[0])  # 各部分不是同一个成分
        # 模糊匹配只容忍拼写错误
        self.assertEqual(ids("Xantham Gum", "Citronelol"), ["xanthan_gum", "citronellol"])
        for name, wrong in [("Laureth-9", "laureth_4"), ("Sodium Myreth Sulfate", "sles"),
                            ("Glyceryl Isostearate", "glyceryl_stearate"), ("Simethicone", "dimethicone")]:
            self.assertNotEqual(getattr(kb.lookup(name)[0], "id", None), wrong, name)

    def test_display_name_and_hints(self):
        from skinsight.kb import ingredient_hints, summarize_flags, user_note
        self.assertEqual(kb.lookup("Ectoin")[0].display, "依克多因（四氢甲基嘧啶羧酸）")
        self.assertEqual(kb.lookup("Glycerin")[0].display, "甘油")  # 没有常用名
        self.assertEqual(kb.lookup("依克多因")[0].id, "ectoin")
        dmdm = kb.lookup("DMDM Hydantoin")[0]
        self.assertIn("甲醛释放体", ingredient_hints(dmdm))
        self.assertTrue(any("甲醛释放体" in m for _, m in summarize_flags(kb.parse("Water, DMDM Hydantoin"))))
        # 给维护者看的出处、审核记录不展示给用户
        self.assertNotRegex(user_note(kb.lookup("p-Aminophenol")[0]), "审核意见|待核实|目录")

    def test_tags_used_by_conflicts(self):
        tags = lambda n: set(kb.lookup(n)[0].tags)
        self.assertIn("bha", tags("Capryloyl Salicylic Acid"))
        self.assertIn("retinoid", tags("Retinal"))
        self.assertIn("fragrance_allergen", tags("Hexyl Cinnamal"))
        self.assertIn("sunscreen", tags("Ethylhexyl Methoxycinnamate"))
        self.assertFalse({"bha", "aha"} & tags("Ethylhexyl Salicylate"))  # 防晒剂，不是酸
        self.assertNotIn("vitamin_c_pure", tags("Sorbic Acid"))
        f = check_routine([("A", kb.parse("Water, Retinal")), ("B", kb.parse("Water, Capryloyl Salicylic Acid"))])
        self.assertEqual(f[0].level, "high")


class TestConflicts(unittest.TestCase):
    def test_retinoid_plus_acid_is_high(self):
        f = check_routine([("A", kb.parse("水, 视黄醇")), ("B", kb.parse("水, 水杨酸"))])
        self.assertEqual(f[0].level, "high")

    def test_gentle_routine_clean(self):
        self.assertEqual(check_routine([("A", kb.parse("水, 甘油")), ("B", kb.parse("水, 泛醇"))]), [])


class TestData(unittest.TestCase):
    def test_synthetic_and_vocab(self):
        s = make_synthetic(kb, n=200)
        self.assertEqual(len(s[0][1]), len(LABELS))
        v = Vocab.build([t for t, _ in s], always=list(kb.items))
        self.assertEqual(v.encode([]), [1])
        self.assertEqual(Vocab.from_dict(v.to_dict()).itos, v.itos)

    def test_split_is_deterministic_and_disjoint(self):
        s = [([str(i)], [0] * 5) for i in range(100)]
        tr, va, te = split_samples(s, seed=7)
        self.assertEqual((len(tr), len(va), len(te)), (80, 10, 10))
        self.assertEqual(split_samples(s, seed=7)[2], te)
        self.assertEqual(len({t[0][0] for t in tr + va + te}), 100)

    def test_auc(self):
        y = [[1] * 5, [0] * 5, [1] * 5, [0] * 5]
        self.assertEqual(auc_scores(y, [[0.9] * 5, [0.1] * 5, [0.8] * 5, [0.2] * 5]), [1.0] * 5)
        self.assertEqual(auc_scores(y, [[1] * 5] * 4), [0.5] * 5)  # 全预测为正

    def test_multiclass_scores(self):
        m = multiclass_scores([0, 1, 2, 2], [0, 1, 2, 1], 3)
        self.assertEqual(m["acc"], 0.75)
        self.assertAlmostEqual(m["f1"][2], 2 / 3)

    def test_kb_features(self):
        v = Vocab(["<pad>", "<unk>", "raw:foo", "parfum", "water"])
        names = kb_feature_names(kb)
        f = kb_feature_matrix(v, kb)
        self.assertEqual(len(f), len(v)); self.assertEqual(len(f[0]), len(names))
        self.assertEqual(f[0], [0.0] * len(names))  # pad
        self.assertEqual(f[2], [0.0] * len(names))  # 未识别成分
        self.assertEqual(f[3][names.index("fragrance")], 1.0)

    def test_cosmetics_csv_tasks(self):
        rows = [["Moisturizer", "Water, Glycerin, Squalane", 1, 1, 0, 0, 1],
                ["Cleanser", "Water, Glycerin, Fragrance", 0, 0, 0, 0, 0],  # 肤质未标注
                ["Cleanser", "Water, Glycerin, Squalane", 1, 0, 0, 0, 0],   # 与第 1 行成分表重复
                ["Face Mask", "No Info", 1, 1, 1, 1, 1]]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.csv")
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(["Label", "Ingredients", *LABELS]); w.writerows(rows)
            skin = load_cosmetics_csv(p, kb)
            cat = load_cosmetics_csv(p, kb, task="category")
        self.assertEqual([y for _, y in skin], [[1.0, 1.0, 0.0, 0.0, 1.0]])
        self.assertEqual([y for _, y in cat], [CATEGORIES.index("Moisturizer"), CATEGORIES.index("Cleanser")])


class TestClean(unittest.TestCase):
    def test_placeholders_and_marketing_dropped(self):
        for t in ["No Info", "#NAME?", "Visit the Dior boutique", "-Vitamin E: Antioxidant. -Clay: Mattifies."]:
            self.assertIsNone(clean_ingredients(t))

    def test_cosmetics_noise(self):
        self.assertEqual(
            clean_ingredients("Avobenzone 3.0%, Octocrylene 5.0%Water, Lemon Oil*, Phenoxyethanol. *Organic. **Natural."),
            "Avobenzone, Octocrylene, Water, Lemon Oil*, Phenoxyethanol")
        self.assertEqual(clean_ingredients("Crm Division: El (Estee Lauder)Ingredients: Water , 1,2-Hexanediol"),
                         "Water, 1,2-Hexanediol")
        self.assertEqual(clean_ingredients("-Hyaluronic Acid: Hydrates. Complex: Water, Squalane"), "Water, Squalane")
        self.assertEqual(clean_ingredients("Organic Argan Oil*. *Organic."), "Organic Argan Oil*")

    def test_chinese_untouched(self):
        self.assertEqual(clean_ingredients("水、甘油、1,2-己二醇，香精"), "水、甘油、1,2-己二醇，香精")


class TestApp(unittest.TestCase):
    """网页 demo 的逻辑部分（不需要 gradio）。"""

    def test_analyze(self):
        import app
        summary, people, marked, rows, sugg = app.analyze(
            "Water, Alcohol Denat., Ectoin, DMDM Hydantoin, 油酸, 胆笛醇, Totally Unknown Thing")
        self.assertIn("知识库已解读 **4** 个，原料目录登记但暂无解读 **1** 个，未识别 **2** 个", summary)
        self.assertIn("甲醛释放体", summary)
        self.assertIn("**干皮、敏感肌**", people)
        self.assertNotRegex(people, "适合|放心")
        self.assertEqual(rows[2][2], "依克多因（四氢甲基嘧啶羧酸）")
        self.assertIn("酒精", rows[1][4])
        self.assertEqual(rows[4][4], "已确认是登记原料，暂无详细解读")  # 油酸：知识库没有，目录里有
        self.assertIn("胆甾醇", rows[5][4])                              # 疑似错字给出候选
        self.assertEqual(rows[6][2], "⚠️ 未识别")
        self.assertEqual(len(rows[0]), len(app.TABLE_HEADERS))
        labels = [lab for _, lab in marked if lab]
        self.assertEqual(labels, ["登记原料，暂无解读", "未识别", "未识别"])
        self.assertIn("胆笛醇 → 胆甾醇", sugg)
        self.assertIn("请先粘贴", app.analyze("  ")[0])

    def test_apply_suggestions_only_what_user_chose(self):
        import app
        text = "水、胆笛醇、工二醇"
        self.assertEqual(app.apply_suggestions(text, []), text)  # 不勾选就不替换
        self.assertEqual(app.apply_suggestions(text, ["胆笛醇 → 胆甾醇", "胆笛醇 → 胆固醇"]), "水、胆甾醇、工二醇")

    def test_check(self):
        import app
        self.assertIn("至少填写 2 个", app.check("A", "水, 视黄醇", "", "", "", "", "", ""))
        out = app.check("A", "水, 视黄醇", "", "水, 水杨酸", "", "", "", "")
        self.assertIn("[高] A + 产品2", out)
        for ex in app.CHECK_EXAMPLES + [[e] for e in app.ANALYZE_EXAMPLES]:
            (app.check if len(ex) == 8 else app.analyze)(*ex)  # 示例都能跑通


    def test_population_notes(self):
        from skinsight.kb import population_notes
        notes = dict(population_notes(kb.parse(
            "Water, Retinol, Glycolic Acid, Ethylhexyl Palmitate, Linalool")))
        self.assertIn("备孕、孕期、哺乳期", notes)
        self.assertIn("屏障受损、医美术后", notes)
        self.assertIn("易长闭口/痘痘者", notes)
        self.assertIn("敏感肌", notes)
        self.assertEqual(population_notes(kb.parse("水、甘油、透明质酸钠")), [])

    @unittest.skipIf(importlib.util.find_spec("rapidocr_onnxruntime") is None, "未安装 rapidocr_onnxruntime")
    def test_recognize_photo_deletes_upload(self):
        import app
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "blank.png")
            cv2.imwrite(p, np.full((60, 200, 3), 255, np.uint8))
            text, status = app.recognize_photo(p, "水、甘油", append=True)
            self.assertFalse(os.path.exists(p))  # 上传的图片不留在磁盘上
        self.assertEqual(text, "水、甘油")
        self.assertIn("没有识别到文字", status)


class TestOCRText(unittest.TestCase):
    """OCR 之后的文字处理（不需要 OCR 模型）。"""

    def test_extract_and_join(self):
        from skinsight.ocr import extract_ingredients
        text, found = extract_ingredients(
            ["产品名称：保湿精华", "全成分：水、甘油、丁二醇、烟酰", "胺、透明质酸钠", "1,2-己二醇、苯氧", "乙醇、香精",
             "使用方法：取适量涂抹", "保质期：3年"], kb)
        self.assertTrue(found)
        self.assertEqual(text, "水、甘油、丁二醇、烟酰胺、透明质酸钠、1,2-己二醇、苯氧乙醇、香精")
        text, _ = extract_ingredients(
            ["INGREDIENTS:Water,ButyleneGlycol,PEG-", "100 Stearate, Caprylic/Capric Trigly-", "ceride, SodiumHydro",
             "xide, Niacinamide", "Parfum.", "Directions: Apply to face."], kb)
        self.assertEqual(text, "Water, Butylene Glycol, PEG-100 Stearate, Caprylic/Capric Triglyceride, "
                               "Sodium Hydroxide, Niacinamide, Parfum")
        text, found = extract_ingredients(["透明质酸钠、1", "2-己二醇"], kb)
        self.assertFalse(found)
        self.assertEqual(text, "透明质酸钠、1,2-己二醇")

    def test_lines_from_tilted_boxes(self):
        from skinsight.ocr import lines_from_ocr
        # 倾斜约 5°：每行被切成左右两个框；第二行左框的 y 比第一行右框还小，不能按原始 y 排序
        s = math.tan(math.radians(5))
        def box(x, y, w=200, h=20):
            return [[x, y + x * s], [x + w, y + (x + w) * s], [x + w, y + h + (x + w) * s], [x, y + h + x * s]]
        result = [[box(0, 0), "A1", 0.9], [box(400, 0), "A2", 0.9], [box(0, 30), "B1", 0.9], [box(400, 30), "B2", 0.9]]
        self.assertEqual(lines_from_ocr(result), ["A1 A2", "B1 B2"])


class TestScreenshotRegression(unittest.TestCase):
    """两张电商详情页截图的回归测试。

    tests/fixtures/ocr_*.json 是 RapidOCR 对截图的原始输出（文字框坐标 + 文字），测试不需要图片和 OCR 模型。
    图片本身有版权，不放进仓库；本地有 data/ocr_samples/*.jpg 时 test_real_ocr 会从图片重新跑一遍。
    """

    @classmethod
    def setUpClass(cls):
        from skinsight.iecic import Catalog
        cls.catalog = Catalog()

    def extract(self, name, result=None):
        import json
        from skinsight.ocr import extract_ingredients, lines_from_ocr
        if result is None:
            path = os.path.join(os.path.dirname(__file__), "fixtures", f"ocr_{name}.json")
            with open(path, encoding="utf-8") as f:
                result = json.load(f)
        text, found = extract_ingredients(lines_from_ocr(result), kb, self.catalog)
        self.assertTrue(found)
        return text

    def check_augustinus(self, text):
        # 左边瓶身上的品牌名、AB 标志不能混进成分表
        self.assertNotRegex(text, "Augustinus|Bader|NB|AB")
        # 小标题不粘到前一个成分上，保留成统一的微量标记
        self.assertIn("向日葵（HELIANTHUS ANNUUS）籽油、其他微量成分：乙醇、", text)
        self.assertEqual(text.count("微量成分"), 1)
        self.assertTrue(text.startswith("水、辛酸/"))
        self.assertTrue(text.endswith("水蒜芥（SISYMBRIUM IRIO）籽油、寡肽-4"))
        for name in ["牛油果树（BUTYROSPERMUM PARKII）果脂", "视黄醇棕榈酸酯", "棕榈酸、神经酰胺"]:
            self.assertIn(name, text)
        items = kb.parse(text)
        self.assertEqual(len(items), 46)
        self.assertIn("retinyl_palmitate", {it.ingredient.id for it in items if it.ingredient})
        from skinsight.kb import population_notes
        self.assertIn("视黄醇棕榈酸酯", dict(population_notes(items))["备孕、孕期、哺乳期"])
        # 知识库和原料目录都对不上的只剩 OCR 认错的「胆笛醇」（胆甾醇）
        unknown = [it.raw for it in items if not it.ingredient and not self.catalog.lookup(it.raw)]
        self.assertEqual(unknown, ["胆笛醇"])
        self.assertEqual(self.catalog.suggest("胆笛醇", kb=kb)[0], "胆甾醇")

    def test_augustinus_bader(self):
        self.check_augustinus(self.extract("augustinus_bader"))

    def test_augustinus_trace_ingredients(self):
        """「其他微量成分」之后（≤0.1%）的成分：活性成分说明、叠用降级、孕期提示措辞、表格标记。"""
        import app
        from skinsight.conflicts import check_routine
        from skinsight.kb import population_notes, summarize_flags
        items = kb.parse(self.extract("augustinus_bader"))
        trace = [it.raw for it in items if it.trace]
        self.assertEqual(trace[0], "乙醇")                       # 标记之后才算微量
        self.assertFalse(any(it.trace for it in items if it.raw == "向日葵（HELIANTHUS ANNUUS）籽油"))
        self.assertEqual(len(trace), 29)
        ret = next(it for it in items if it.ingredient and it.ingredient.id == "retinyl_palmitate")
        self.assertTrue(ret.trace)
        # 1. 活性成分提示
        msgs = [m for _, m in summarize_flags(items)]
        self.assertNotIn("主要活性成分", "".join(msgs))
        self.assertIn("含有的活性成分：维A类、纯VC、VC衍生物（均为微量成分，≤0.1%，作用可能有限）", msgs)
        # 2. 叠用检查：微量成分触发的冲突降一级，并注明
        f = check_routine([("AB乳霜", items), ("果酸水", kb.parse("水、乙醇酸、甘油"))])
        hit = next(x for x in f if x.message.startswith("维A类 + 酸类"))
        self.assertEqual(hit.level, "medium")                      # 原本是 high
        self.assertIn("AB乳霜中的维A类只是微量成分，≤0.1%", hit.message)
        f2 = check_routine([("A醇精华", kb.parse("水、视黄醇")), ("果酸水", kb.parse("水、乙醇酸、甘油"))])
        self.assertEqual(f2[0].level, "high")                      # 不是微量的照旧
        # 3. 需注意人群：微量维A类保留提示，换措辞
        notes = dict(population_notes(items))
        self.assertTrue(notes["备孕、孕期、哺乳期"].startswith("含微量维A类（≤0.1%），备孕、孕期、哺乳期建议咨询医生"))
        # 屏障受损、医美术后：只由微量的纯VC触发，保留提示并注明微量，措辞与孕期提示一致
        self.assertTrue(notes["屏障受损、医美术后"].startswith("含微量纯VC（≤0.1%），屏障受损、医美术后建议咨询医生"))
        # 表格：微量成分在提示列标出
        rows = app.analyze(self.extract("augustinus_bader"))[3]
        row = next(r for r in rows if r[1] == "视黄醇棕榈酸酯")
        self.assertTrue(row[4].startswith("微量（≤0.1%）"))
        self.assertNotIn("微量", next(r for r in rows if r[1] == "甘油")[4])

    def test_no_trace_header_keeps_old_logic(self):
        from skinsight.conflicts import check_routine
        from skinsight.kb import population_notes, summarize_flags
        items = kb.parse("水、甘油、视黄醇棕榈酸酯、抗坏血酸")
        self.assertFalse(any(it.trace for it in items))
        self.assertIn("含有的活性成分：维A类、纯VC", [m for _, m in summarize_flags(items)])
        self.assertTrue(dict(population_notes(items))["备孕、孕期、哺乳期"].startswith("含维A类"))
        self.assertEqual(check_routine([("A", items), ("B", kb.parse("水、乙醇酸"))])[0].level, "high")
        self.assertTrue(dict(population_notes(items))["屏障受损、医美术后"].startswith("含纯VC（"))
        # 粘贴的成分表里自带「其他微量成分：」也一样识别
        self.assertTrue(kb.parse("水、甘油、其他微量成分：视黄醇")[-1].trace)
        # 微量和非微量混合：保留原措辞，微量的那个单独标注
        mixed = dict(population_notes(kb.parse("水、乙醇酸、其他微量成分：抗坏血酸")))["屏障受损、医美术后"]
        self.assertIn("抗坏血酸（维生素C）（微量）", mixed)
        self.assertIn("慎用；术后护理请遵医嘱", mixed)

    def test_curel_mask(self):
        text = self.extract("curel_mask")
        # 左边红色标注「净含量」不能截断成分表；「。」「.」认错的逗号按分隔符处理
        self.assertNotIn("净含量", text)
        self.assertTrue(text.startswith("水、甘油、"))
        self.assertTrue(text.endswith("EDTA二钠、精氨酸、羟丙基甲基纤维素、蓝桉叶提取物"))
        sugg = self.catalog.suggest("甘草酸二钾丙烯酸（酯）类/C10-30烷醇丙烯酸酯交联聚合物", kb=kb)
        self.assertTrue(sugg[0].startswith("甘草酸二钾、丙烯酸（酯）类"))  # 漏掉的逗号：给出拆分建议

    @unittest.skipUnless(os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "ocr_samples",
                                                     "augustinus_bader.jpg"))
                         and importlib.util.find_spec("rapidocr_onnxruntime"), "没有本地截图或 OCR")
    def test_real_ocr(self):
        from skinsight.ocr import image_from_bytes, _ENGINE  # noqa: F401
        from rapidocr_onnxruntime import RapidOCR
        path = os.path.join(os.path.dirname(__file__), "..", "data", "ocr_samples", "augustinus_bader.jpg")
        with open(path, "rb") as f:
            result, _ = RapidOCR()(image_from_bytes(f.read()))
        self.check_augustinus(self.extract("augustinus_bader", result))


class TestCatalog(unittest.TestCase):
    def test_lookup_and_suggest(self):
        from skinsight.iecic import Catalog
        c = Catalog()
        self.assertGreater(len(c), 8000)
        self.assertEqual(c.lookup("野大豆（GLYCINESOJA）蛋白").no, 7609)    # 括号里拉丁名少了空格
        self.assertEqual(c.lookup("牛油果树果脂").no, 4891)                  # 只写中文部分
        self.assertEqual(c.lookup("OLEIC ACID").zh, "油酸")
        self.assertIsNone(c.lookup("完全不存在的成分"))
        self.assertIn("氢氧化钠", c.suggest("氧氧化钠", kb=kb))

    def test_kb_chinese_only_and_slash_mixture(self):
        self.assertEqual(kb.lookup("牛油果树果脂")[0].id, "shea_butter")
        self.assertEqual(kb.lookup("牛油果树（BUTYROSPERMUMPARKII）果脂")[0].id, "shea_butter")
        ids = [it.ingredient.id for it in kb.parse("神经酰胺 NS/神经酰胺NG、甘油")]
        self.assertEqual(ids, ["ceramide_ns", "ceramide_ng", "glycerin"])
        self.assertEqual(kb.lookup("Caprylic/Capric Triglyceride")[0].id, "caprylic_capric_tg")  # 本身带斜杠的不拆


@unittest.skipIf(torch is None, "未安装 torch")
class TestModel(unittest.TestCase):
    def test_forward_and_training_step(self):
        from skinsight.model import SkinTypeNet
        from skinsight.train import collate
        m = SkinTypeNet(vocab_size=50, num_labels=5, max_len=16)
        x, y = collate([([3, 4, 5], [1, 0, 1, 0, 1]), ([6], [0, 1, 0, 1, 0])])
        logits, attn = m(x)
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.allclose(attn.sum(-1), torch.ones(2)))
        self.assertTrue(torch.all(attn[1, 1:] == 0))  # padding 不分配注意力
        torch.nn.functional.binary_cross_entropy_with_logits(logits, y).backward()

    def test_kb_feats_and_category(self):
        from skinsight.model import SkinTypeNet
        from skinsight.train import collate
        m = SkinTypeNet(vocab_size=10, num_labels=6, max_len=16, kb_feat_dim=3)
        m.kb_feats.copy_(torch.rand(10, 3))
        x, y = collate([([3, 4], 2), ([5], 0)])
        self.assertEqual(y.dtype, torch.long)
        logits, _ = m(x)
        torch.nn.functional.cross_entropy(logits, y).backward()
        self.assertIsNotNone(m.kb_proj.weight.grad)
        # kb_feats 随 state_dict 保存；旧 checkpoint 的 config 没有 kb_feat_dim 也能加载
        m2 = SkinTypeNet(**m.config); m2.load_state_dict(m.state_dict())
        self.assertTrue(torch.equal(m2.kb_feats, m.kb_feats))
        old = SkinTypeNet(vocab_size=10, max_len=16).config; old.pop("kb_feat_dim")
        SkinTypeNet(**old)



@unittest.skipIf(torch is None, "未安装 torch")
class TestPretrain(unittest.TestCase):
    def test_mask_tokens(self):
        from skinsight.pretrain import IGNORE, mask_tokens
        x = torch.tensor([[5, 6, 7, 8, 0, 0], [1, 9, 0, 0, 0, 0]])  # 第 2 行只有 1 个可遮（1 是 unk）
        inp, tgt = mask_tokens(x, mask_id=2, vocab_size=20, gen=torch.Generator().manual_seed(0))
        m = tgt != IGNORE
        self.assertTrue(m.any(1).all())                       # 每行至少遮 1 个
        self.assertFalse(m[x <= 2].any())                     # pad / unk 不作为目标
        self.assertTrue(torch.equal(tgt[m], x[m]))            # 目标是原 token
        self.assertTrue(torch.equal(inp[~m], x[~m]))          # 未选中的位置不变

    def test_mlm_and_encoder_transfer(self):
        from skinsight.model import MaskedIngredientModel, SkinTypeNet
        mlm = MaskedIngredientModel(vocab_size=30, d_model=16, max_len=8)
        self.assertEqual(tuple(mlm(torch.tensor([[3, 4, 5]])).shape), (1, 3, 30))
        net = SkinTypeNet(vocab_size=30, num_labels=6, d_model=16, max_len=8)
        net.load_encoder(mlm.net.state_dict())
        self.assertTrue(torch.equal(net.tok.weight, mlm.net.tok.weight))
        self.assertFalse(torch.equal(net.head[1].weight[:5], mlm.net.head[1].weight[:5]))  # 分类头不迁移
        with self.assertRaises(ValueError):
            SkinTypeNet(vocab_size=31, d_model=16, max_len=8).load_encoder(mlm.net.state_dict())

    def test_build_vocab(self):
        from skinsight.data_utils import MASK
        from skinsight.pretrain import build_vocab
        v = build_vocab([["raw:a", "raw:a", "raw:b"]], [["raw:c", "raw:c"]], kb, min_freq=2)
        self.assertEqual(v.itos[:3], ["<pad>", "<unk>", MASK])
        self.assertIn("raw:a", v.stoi); self.assertIn("raw:c", v.stoi); self.assertNotIn("raw:b", v.stoi)
        self.assertIn("niacinamide", v.stoi)


if __name__ == "__main__":
    unittest.main()

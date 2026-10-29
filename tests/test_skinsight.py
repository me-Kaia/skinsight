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

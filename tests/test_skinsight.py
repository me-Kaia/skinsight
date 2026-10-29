import unittest

from skinsight.conflicts import check_routine
from skinsight.data_utils import (LABELS, Vocab, auc_scores, clean_ingredients, make_synthetic,
                                  split_samples, to_tokens)
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


if __name__ == "__main__":
    unittest.main()

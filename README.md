# skinsight 🧴

护肤品成分表解读 + 叠用冲突检查 + 基于 PyTorch 的肤质适配预测。

把成分表（中文或 INCI 英文都行）贴进来，它会告诉你：

- 每个成分是什么、起什么作用（中英文自动对齐，「烟酰胺」和「Niacinamide」是同一个东西）
- 有没有香精、高位酒精、强致敏防腐剂、高致痘评分成分
- 几个产品一起用会不会打架（维A + 酸、VC + 过氧化苯甲酰……），以及哪些是过时的「冲突」说法
- 模型预测这个产品适合哪种肤质，并用注意力权重指出**是哪几个成分**导致了这个判断

## 快速开始

```bash
pip install -r requirements.txt   # 只用成分解读和冲突检查的话，不装 torch 也能跑

# 解读成分表
python -m skinsight analyze "水、甘油、烟酰胺、1,2-己二醇、香精、芳樟醇"

# 检查叠用冲突
python -m skinsight check \
  -p "A醇精华: 水, 视黄醇, 角鲨烷" \
  -p "水杨酸水: 水, 水杨酸, 烟酰胺"
# [高] A醇精华 + 水杨酸水：维A类 + 酸类：同一次叠用刺激明显叠加……建议分早晚或隔天使用。

# 训练模型（先用合成数据跑通流程）
python -m skinsight.train --synthetic --epochs 10
python -m skinsight.baseline --synthetic          # 词袋 + 逻辑回归对比（需要 scikit-learn）
python -m skinsight analyze "Water, Glycerin, Fragrance, Coconut Oil" --ckpt checkpoints/skin_type.pt
```

## 模型

```
成分表 ──解析/归一化──▶ 成分 token
        token embedding + 排序位置 embedding
                 │
        Transformer Encoder ×2
                 │
        Attention Pooling ──▶ 注意力权重（可解释：哪些成分最关键）
                 │
        Linear ──▶ 混合肌 / 干皮 / 中性肌 / 油皮 / 敏感肌（多标签，sigmoid）
```

设计要点：

- **位置有意义**：INCI 成分表按含量从高到低排列（约 1% 以下可任意排序），所以「排第几位」作为位置 embedding 输入，而不是把成分当成无序集合。
- **知识库对齐词表**：知识库收录的成分用统一 id 作为 token，中文和英文成分表推理时落在同一个 embedding 上。
- **类别不平衡**：`BCEWithLogitsLoss` 按正负样本比设置 `pos_weight`，按验证集 macro-F1 选最优 checkpoint。

## 数据

| 来源 | 用法 |
|---|---|
| 合成数据 `--synthetic` | 只用于跑通流程和测试，**不代表真实规律** |
| [Kaggle: Cosmetics datasets](https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets) | 1,472 条丝芙兰产品，带成分表和 5 种肤质适配标签，下载后 `--csv data/cosmetics.csv`（CSV 不入库）；清洗后可用 853 条，见 [Results](#results) |
| 自建 JSONL `--jsonl` | 每行 `{"ingredients": "水, 甘油, ...", "labels": {"Dry": 1, "Sensitive": 1}}` |

知识库在 `skinsight/data/ingredients.json`，目前收录 64 个常见成分，欢迎 PR 补充。

## 项目结构

```
skinsight/
├── data/ingredients.json   成分知识库（中英文名、别名、功效、标签、刺激/致痘等级）
├── kb.py                   成分表解析、别名/括号/模糊匹配、规则提示
├── conflicts.py            叠用冲突规则引擎
├── data_utils.py           token 化、词表、数据集加载、合成数据
├── model.py                SkinTypeNet（PyTorch）
├── train.py                训练脚本
├── baseline.py             词袋 + 逻辑回归 baseline（同划分、同指标）
├── predict.py              推理 + 注意力解释
└── cli.py                  命令行入口
tests/                      单元测试：python -m unittest
```

## Results

在 Kaggle Cosmetics datasets 上的结果。**结论先说：在这份数据上，SkinTypeNet 和词袋逻辑回归都没有超过「全部预测为适合」的平凡基线（macro-F1）；按 ROC-AUC 看，逻辑回归学到了一点信号，SkinTypeNet 更弱。** 主要原因在数据本身，见下。

### 数据

| | 条数 |
|---|---|
| CSV 原始行数 | 1,472 |
| − 成分表是占位符或纯营销文案（"No Info"、"Visit the Dior boutique"、"#NAME?"、「-成分: 功效」描述等） | 211 |
| − 5 个标签全为 0（视为「未标注」，默认丢弃，`--keep-unlabeled` 可保留） | 374 |
| − 清洗后不足 3 个成分 | 7 |
| − 成分表完全相同的重复行（同一产品出现在多个品类，去重防止跨 train/test 泄漏） | 27 |
| **最终可用（默认设置）** | **853**（train/val/test = 683/85/85） |
| 保留全 0 行时（去掉 211 + 11 + 37） | 1,213（971/121/121） |

各标签正样本比例（默认设置 / 保留全 0 行）：

| Combination | Dry | Normal | Oily | Sensitive |
|---|---|---|---|---|
| 97.1% / 68.2% | 90.3% / 63.4% | 96.0% / 67.4% | 89.9% / 63.1% | 76.2% / 53.5% |

- 有标注的产品里 **67% 被标为 5 种肤质全部适合**，负样本很少，这是 F1 难以超过平凡基线的根本原因。
- 全 0 行约占原始数据 1/3，且在 6 个品类里比例相近，更像漏标而非「不适合任何肤质」。保留它们时，任务很大程度上变成「这个产品有没有被标注」，和成分关系不大。
- 知识库（64 个成分）在 token 级别识别了 **27.5%** 的成分，98.8% 的产品至少命中 1 个；没识别的成分以原文 token 进入词表（`raw:caprylyl glycol` 等），模型同样能用。最常见的未收录成分：辛甘醇、生育酚乙酸酯、柠檬酸、山梨酸钾、氢氧化钠。

`load_cosmetics_csv` 的清洗（`clean_ingredients`）：去掉占位符和纯营销文案（营销文案后面接了真实成分表的保留成分表部分）、"...Ingredients:" 前缀、"Please be aware..." 等免责声明、`*Organic.` 类脚注、浓度（含 `5.0%Water` 这种粘连）、`(and)` / `May Contain:` 分隔；然后丢弃全 0 标签行并按成分表去重。

### 指标

同一 `split_samples` 划分，5 个随机种子（1–5）取平均，test 集。阈值 0.5 的 F1 + 与阈值无关的 ROC-AUC（平凡基线的 AUC 恒为 0.5）。

**默认设置（丢弃全 0 行，853 条）**

| 方法 | Combination | Dry | Normal | Oily | Sensitive | macro-F1 | macro-AUC |
|---|---|---|---|---|---|---|---|
| 全部预测为适合 | 0.988 | 0.957 | 0.989 | 0.943 | 0.845 | **0.944 ± 0.002** | 0.500 |
| 词袋 + 逻辑回归 | 0.984 | 0.927 | 0.984 | 0.898 | 0.799 | 0.919 ± 0.004 | **0.666 ± 0.045** |
| SkinTypeNet | 0.978 | 0.924 | 0.973 | 0.910 | 0.814 | 0.920 ± 0.005 | 0.560 ± 0.072 |

**保留全 0 行（1,213 条）**

| 方法 | Combination | Dry | Normal | Oily | Sensitive | macro-F1 | macro-AUC |
|---|---|---|---|---|---|---|---|
| 全部预测为适合 | 0.805 | 0.774 | 0.800 | 0.763 | 0.708 | **0.770 ± 0.029** | 0.500 |
| 词袋 + 逻辑回归 | 0.764 | 0.738 | 0.764 | 0.710 | 0.651 | 0.726 ± 0.029 | **0.703 ± 0.026** |
| SkinTypeNet | 0.787 | 0.749 | 0.775 | 0.721 | 0.674 | 0.741 ± 0.019 | 0.650 ± 0.042 |

- SkinTypeNet：`--epochs 40 --lr 3e-4 --d-model 128`，2 层，按验证集 macro-F1 选 epoch。在种子 42 上试了 lr ∈ {1e-3, 3e-4} × d_model ∈ {32, 64, 128}；验证集最高的一组（lr=1e-3, d=64）最优 epoch 是第 1 轮，实际上退化成了「全预测为正」，所以没选它。
- 逻辑回归：二值词袋（同一套 token 和 min_freq=2），每个标签一个 `class_weight="balanced"` 的模型，C 在验证集上从 {0.01 … 3} 里选。
- test 集只有 85 / 121 条，± 是 5 个种子间的标准差，差距在 0.01 左右的结论都不可靠。
- 用 `pos_weight` / `class_weight` 平衡类别会把一部分本该是正的预测压成负，在正样本占九成的数据上这对 F1 是净亏的；这也是两者 F1 都低于平凡基线的直接原因。

复现：

```bash
python -m skinsight.baseline --csv data/cosmetics.csv --seed 1 --metrics-out runs/lr_1.json
python -m skinsight.train --csv data/cosmetics.csv --seed 1 --epochs 40 --lr 3e-4 --d-model 128 --metrics-out runs/m_1.json
```

### 预测示例

用默认设置、种子 42 训练的 checkpoint（test macro-F1 0.926），下面两个产品都在它的 test 集里，两个都有判错：

**LANEIGE Water Bank Moisture Cream**（面霜）

| | 混合肌 | 干皮 | 中性肌 | 油皮 | 敏感肌 |
|---|---|---|---|---|---|
| 标签 | 0 | 1 | 1 | 0 | 1 |
| 预测 | 0.98 | 1.00 | 0.99 | 0.85 | 1.00 |

注意力最高：生育酚 (0.11)、Hydrogenated Polyisobutene (0.06)、Acetyl Glucosamine (0.06)。注意力很分散，模型没能识别出「偏滋润、不适合油皮」，混合肌和油皮都判错了。

**CAUDALIE Beauty Elixir**（喷雾）

| | 混合肌 | 干皮 | 中性肌 | 油皮 | 敏感肌 |
|---|---|---|---|---|---|
| 标签 | 1 | 0 | 1 | 1 | 1 |
| 预测 | 1.00 | 0.98 | 0.99 | 0.99 | 0.16 |

注意力最高：丁香酚 Eugenol (0.26)、迷迭香叶油 (0.12)、变性乙醇 (0.11)。模型因为香料致敏原和排第 2 位的酒精判它不适合敏感肌，这个理由本身说得通，但和数据标签（标为适合敏感肌）不一致；干皮也判错了。

## Roadmap

- [ ] OCR：拍照识别成分表
- [ ] 根据肤质和所在城市的气候生成护肤流程
- [ ] Web Demo（Gradio）
- [ ] 扩充知识库，对接《已使用化妆品原料目录》

## 免责声明

本项目用于学习和信息参考，不构成医疗或皮肤科诊疗建议。致痘评分来自早期动物实验汇总，仅供参考；成品的实际表现取决于浓度和整体配方。皮肤出现持续不适请咨询医生。

## License

MIT

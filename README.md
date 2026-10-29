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
| [Kaggle: Cosmetics datasets](https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets) | 约 1,400 条丝芙兰产品，带成分表和 5 种肤质适配标签，下载后 `--csv data/cosmetics.csv` |
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
├── predict.py              推理 + 注意力解释
└── cli.py                  命令行入口
tests/                      单元测试：python -m unittest
```

## Roadmap

- [ ] OCR：拍照识别成分表
- [ ] 根据肤质和所在城市的气候生成护肤流程
- [ ] Web Demo（Gradio）
- [ ] 扩充知识库，对接《已使用化妆品原料目录》

## 免责声明

本项目用于学习和信息参考，不构成医疗或皮肤科诊疗建议。致痘评分来自早期动物实验汇总，仅供参考；成品的实际表现取决于浓度和整体配方。皮肤出现持续不适请咨询医生。

## License

MIT

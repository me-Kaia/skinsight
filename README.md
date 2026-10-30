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

## 网页 Demo

用 Gradio 做的中文网页，两个标签页：

- **成分解读**：粘贴成分表，或上传/拍摄成分表照片自动识别文字（识别结果填进可编辑文本框，核对后点「分析」）。显示每个成分的名称、作用和提示，以及「需注意人群」：香精/致敏原 → 敏感肌慎用；致痘评分 ≥ 4 → 易长闭口/痘痘者留意；维A类 → 备孕、孕期、哺乳期一般建议避免；酸类、纯VC、高刺激成分 → 屏障受损、医美术后慎用；甲醛释放体 → 敏感肌慎用；高位酒精 → 干皮、敏感肌留意。只做基于成分规则的提示，不做肤质预测。
- **叠用检查**：2–4 个产品的冲突和建议。

```bash
pip install "gradio>=4.44,<6" "rapidocr_onnxruntime==1.2.3"   # 不装 OCR 也能用，只是不能识别照片
python app.py        # 打开 http://127.0.0.1:7860
```

如果开着系统代理（Clash 等）时启动报 `startup-events failed (code 502)`，先设置 `NO_PROXY=localhost,127.0.0.1` 再启动。

### 拍照识别

- OCR 用 [RapidOCR](https://github.com/RapidAI/RapidOCR)（`rapidocr_onnxruntime`，PP-OCRv3 中英文模型），在本地 CPU 上运行，模型随 pip 包安装，不需要 API key，单张图约 2 秒。
- 上传的图片读进内存识别，识别后立即删除 Gradio 的上传缓存文件，并清空图片组件；`delete_cache` 每 10 分钟再兜底清理一次。
- 识别后自动截取「全成分 / 成分 / Ingredients」之后、「使用方法 / 保质期 / Directions」之前的内容；按倾角把文字框排回阅读顺序；把折行断开的成分名拼回去（中文直接拼接，英文补空格、去掉断词连字符，用知识库判断换行处是同一个成分还是漏识别的逗号）；OCR 丢了空格的英文成分名（`ButyleneGlycol`）按知识库换回标准写法。

在合成测试图上的结果（用 Windows 自带字体渲染的成分表，加噪声、模糊、旋转、圆柱形弯曲和反光模拟拍照；不是真实照片，实际效果通常会差一些）：

| 测试图 | 成分完全正确 | 知识库能对上 | 主要问题 |
|---|---|---|---|
| 中文、平拍 | 20/20 | 20/20 | — |
| 英文、平拍、小字 | 22/23 | 23/23 | `Ethylhexylglycerin` 认成 `Ethyihexylglycerin`（l→i），知识库模糊匹配能纠正 |
| 中文、小字、模糊、偏暗、倾斜 3° | 12/14 | 12/14 | 形近字：氢→氧（氢氧化钠）、育→裔（生育酚） |
| 瓶身弯曲、文字行接近水平 | 3/16 | 4/16 | 两侧压缩 + 反光处丢字母（`Ethylhexyl` → `Ehyhexy`），逗号丢失导致几个成分粘在一起 |
| 瓶身弯曲、文字行明显成弧形 | 0/16 | 0/16 | 检测模型找不到正文 |

平拍的标签基本可用；弯曲瓶身是主要短板，页面上提示让文字行保持水平、分段拍摄，并支持「追加到已有文本」逐段识别。知识库认不出的成分会在状态栏提醒用户核对。

**真实截图**（电商详情页，左边是商品图、右边是成分表；回归测试 `TestScreenshotRegression`，OCR 原始输出存在 `tests/fixtures/`，截图本身有版权不入库）：

- 版面分栏：找到「成分」标题所在的文字框，只保留和它横向重叠的文字，瓶身上的品牌名、标志不会再混进成分表；左侧图片里的「净含量」标注也不会把成分表截断。
- 「其他微量成分：」这类小标题当分隔符；括号里的拉丁名去掉换行处的连字符、恢复空格（`牛油果树（BUTYROSPE-` + `RMUMPARKII）果脂` → `牛油果树（BUTYROSPERMUM PARKII）果脂`），中文名只写中文部分也能匹配；中文后面被认成句点的逗号按分隔符处理；「神经酰胺 NS/神经酰胺 NG」这类斜杠并列按两个成分处理。
- 奥古斯汀·巴德乳霜：46 个成分完整截出，视黄醇棕榈酸酯识别为维A类、「需注意人群」出现孕期提示；知识库和原料目录都对不上的只剩 OCR 认错的「胆笛醇」，候选第一个就是「胆甾醇」。
- 珂润面膜：15 个成分完整截出；3 个认错的（工二醇、棕桐酰肢、漏掉逗号粘在一起的「甘草酸二钾丙烯酸（酯）类…」）都给出了正确候选，包括拆成两个成分的建议。

**原料目录备用字典**：知识库只解读几百个常见成分，其余成分用《已使用化妆品原料目录（2021年版）》（`skinsight/data/iecic_2021.tsv`，8,965 条，只含序号、标准中文名、INCI 名）确认：目录里有的显示「已确认是登记原料，暂无详细解读」；都对不上的标红，并给出目录里最接近的名字（字形相近、或者漏了分隔符的拆分建议）让用户勾选确认，不会自动替换。该目录是国家药监局 2021 年第 62 号公告的附件，属于行政规范性文件（《著作权法》第五条不适用著作权保护的范围），随项目分发并注明出处；2025 年国家药监局已将其调整为目录Ⅰ、Ⅱ，这里只用于识别成分名称。

页面上还可以先**裁剪**照片（上传后点编辑器左侧的图片按钮 → Crop，只框住成分表）再识别；分析结果用颜色标出未识别（红）和目录登记但暂无解读（黄）的成分。

### 部署到魔搭创空间（ModelScope）

`deploy/modelscope/` 里是创空间需要的卡片 `README.md`（YAML 头指定入口 `app.py`）、`requirements.txt`（Gradio + RapidOCR）、`packages.txt`（opencv 在 Linux 上需要的 `libgl1`、`libglib2.0-0`），`prepare.py` 负责把 demo 用到的文件（`app.py`、知识库、规则引擎、OCR 文字处理、原料目录，不含肤质模型和训练代码）复制到一个目录。在 Linux + Python 3.11 的干净环境里按这份 requirements 安装并运行过拍照识别。

网页上传：

1. 在 [魔搭社区](https://www.modelscope.cn/) 登录后「创建创空间」，接入 SDK 选 **Gradio**，云资源选免费 CPU 即可，镜像用默认的最新版本；
2. `python deploy/modelscope/prepare.py D:\skinsight-upload` 生成要上传的文件；
3. 在「创空间详情 → 空间文件」点「添加文件 → 上传文件」，保持目录结构上传（`skinsight/`、`skinsight/data/` 两层子目录不能打平）；
4. 点「立即发布」（或设置页的「上线」）。之后更新代码：重新生成、上传覆盖改动过的文件，再点设置页的「重启创空间」。

也可以用 git：`git clone http://oauth2:<your_access_token>@www.modelscope.cn/studios/<your_name>/<studio_name>.git`，`prepare.py` 复制到克隆下来的目录后 `git add -A && git commit && git push`。

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
- **可选知识库特征**（`--kb-feats`）：刺激/致痘等级和 tags 投影后加到 token embedding 上；实测没有可靠提升，默认关闭，见 [Results](#results)。
- **可选预训练**（`--init-ckpt`）：在 Open Beauty Facts 无标签成分表上做 masked ingredient modeling，再微调；对类别任务有帮助，对肤质任务没有可靠提升，见 [Pretraining](#pretraining)。
- **类别不平衡**：`BCEWithLogitsLoss` 按正负样本比设置 `pos_weight`，按验证集 macro-F1 选最优 checkpoint。

## 数据

| 来源 | 用法 |
|---|---|
| 合成数据 `--synthetic` | 只用于跑通流程和测试，**不代表真实规律** |
| [Kaggle: Cosmetics datasets](https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets) | 1,472 条丝芙兰产品，带成分表和 5 种肤质适配标签，下载后 `--csv data/cosmetics.csv`（CSV 不入库）；清洗后可用 853 条，见 [Results](#results) |
| [Open Beauty Facts](https://world.openbeautyfacts.org/)（ODbL） | 1.4 万条无标签成分表，只用于自监督预训练，放在 `data/openbeautyfacts/`（不入库），见 [Pretraining](#pretraining) |
| 自建 JSONL `--jsonl` | 每行 `{"ingredients": "水, 甘油, ...", "labels": {"Dry": 1, "Sensitive": 1}}` |

知识库在 `skinsight/data/ingredients.json`，当前版本 v0.2 收录 377 个成分（v0.1 为 64 个），按 Kaggle 数据和 Open Beauty Facts 语料中的出现频率补充。中文名（`zh`）以《已使用化妆品原料目录（2021年版）》标准名为准，常用叫法放在 `aliases`，少数成分另有用于显示的常用名 `common_zh`（显示为「依克多因（四氢甲基嘧啶羧酸）」）；已核实的刺激/致痘等级在 `note` 里写明依据，没把握的填 0 并标注「待核实」。欢迎 PR 补充。下面的实验结果基于 v0.1，见 [Results](#results)。

## 项目结构

```
skinsight/
├── data/ingredients.json   成分知识库（中英文名、别名、功效、标签、刺激/致痘等级）
├── data/iecic_2021.tsv     《已使用化妆品原料目录（2021年版）》序号、中文名、INCI
├── kb.py                   成分表解析、别名/括号/模糊匹配、规则提示
├── conflicts.py            叠用冲突规则引擎
├── data_utils.py           token 化、词表、数据集加载、合成数据
├── model.py                SkinTypeNet、MaskedIngredientModel（PyTorch）
├── train.py                训练脚本（--init-ckpt 加载预训练 encoder）
├── pretrain_data.py        从 Open Beauty Facts 导出准备预训练语料
├── pretrain.py             masked ingredient modeling 自监督预训练
├── baseline.py             词袋 + 逻辑回归 baseline（同划分、同指标）
├── predict.py              推理 + 注意力解释
├── ocr.py                  拍照识别：OCR 文字排序、截取成分段、拼接断行
├── iecic.py                《已使用化妆品原料目录》备用字典：登记原料确认、疑似错字候选
└── cli.py                  命令行入口
app.py                      网页 demo（Gradio）
deploy/modelscope/          魔搭创空间部署文件
tests/                      单元测试：python -m unittest discover -s tests
```

## Results

在 Kaggle Cosmetics datasets 上的结果。**结论先说：在这份数据上，SkinTypeNet 和词袋逻辑回归都没有超过「全部预测为适合」的平凡基线（macro-F1）；按 ROC-AUC 看，逻辑回归学到了一点信号，SkinTypeNet 更弱，预训练也没有改变这个排序（见 [Pretraining](#pretraining)）。** 主要原因在数据本身，见下。

> **知识库版本**：本节所有实验结果（包括识别率、各表格和预测示例）都是基于旧版知识库 **v0.1**（64 个成分）得出的，知识库扩充到 v0.2 后没有重跑。

> **切分 bug 修复**：早期版本把英文逗号的切分规则写成了「逗号任意一侧是数字就不切」，本意是保护 `1,2-Hexanediol`，结果 `Polyquaternium-7, Parfum`、`Steareth-10,Dimethicone` 这类成分也被粘成一个 token（约 4% 的 token、66% 的丝芙兰产品受影响）。现在只有逗号**两侧都是数字**时才不切，并补了测试。下面所有数字都是修复后从头重跑的结果（包括预训练），和之前提交里的数字不可比。

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
- 知识库（64 个成分）在 token 级别识别了 **27.5%** 的成分，98.9% 的产品至少命中 1 个；没识别的成分以原文 token 进入词表（`raw:caprylyl glycol` 等），模型同样能用。最常见的未收录成分：辛甘醇、柠檬酸、生育酚乙酸酯、山梨酸钾、氢氧化钠。

`load_cosmetics_csv` 的清洗（`clean_ingredients`）：去掉占位符和纯营销文案（营销文案后面接了真实成分表的保留成分表部分）、"...Ingredients:" 前缀、"Please be aware..." 等免责声明、`*Organic.` 类脚注、浓度（含 `5.0%Water` 这种粘连）、`(and)` / `May Contain:` 分隔；然后丢弃全 0 标签行并按成分表去重。

### 指标

同一 `split_samples` 划分，5 个随机种子（1–5）取平均，test 集。阈值 0.5 的 F1 + 与阈值无关的 ROC-AUC（平凡基线的 AUC 恒为 0.5）。默认设置下有一个种子的 test 集里 Combination 没有负样本，该标签的 AUC 只在 4 个种子上平均。

**默认设置（丢弃全 0 行，853 条）**

| 方法 | Combination | Dry | Normal | Oily | Sensitive | macro-F1 | macro-AUC |
|---|---|---|---|---|---|---|---|
| 全部预测为适合 | 0.988 | 0.957 | 0.989 | 0.943 | 0.845 | **0.944 ± 0.002** | 0.500 |
| 词袋 + 逻辑回归 | 0.983 | 0.921 | 0.984 | 0.898 | 0.807 | 0.919 ± 0.007 | **0.679 ± 0.061** |
| SkinTypeNet | 0.979 | 0.935 | 0.976 | 0.905 | 0.796 | 0.918 ± 0.008 | 0.619 ± 0.038 |

**保留全 0 行（1,213 条）**

| 方法 | Combination | Dry | Normal | Oily | Sensitive | macro-F1 | macro-AUC |
|---|---|---|---|---|---|---|---|
| 全部预测为适合 | 0.805 | 0.774 | 0.800 | 0.763 | 0.708 | **0.770 ± 0.029** | 0.500 |
| 词袋 + 逻辑回归 | 0.774 | 0.745 | 0.768 | 0.712 | 0.670 | 0.734 ± 0.036 | **0.702 ± 0.032** |
| SkinTypeNet | 0.776 | 0.743 | 0.763 | 0.725 | 0.684 | 0.738 ± 0.051 | 0.641 ± 0.040 |

- SkinTypeNet：`--epochs 40 --lr 1e-3 --d-model 128`，2 层，按验证集 macro-F1 选 epoch。在种子 42 上试了 lr ∈ {1e-3, 3e-4} × d_model ∈ {32, 64, 128}（30 epoch），选验证集最高的一组；最优 epoch 为第 1 轮的配置（实际上退化成「全预测为正」）不参与选择，这一轮没有出现。
- 逻辑回归：二值词袋（同一套 token 和 min_freq=2），每个标签一个 `class_weight="balanced"` 的模型，C 在验证集上从 {0.01 … 3} 里选。
- test 集只有 85 / 121 条，± 是 5 个种子间的标准差，差距在 0.01 左右的结论都不可靠。
- 用 `pos_weight` / `class_weight` 平衡类别会把一部分本该是正的预测压成负，在正样本占九成的数据上这对 F1 是净亏的；这也是两者 F1 都低于平凡基线的直接原因。

复现：

```bash
python -m skinsight.baseline --csv data/cosmetics.csv --seed 1 --metrics-out runs/lr_1.json
python -m skinsight.train --csv data/cosmetics.csv --seed 1 --epochs 40 --lr 1e-3 --d-model 128 --metrics-out runs/m_1.json
```

### 预测示例

用默认设置、种子 42 训练的 checkpoint（test macro-F1 0.925），下面两个产品都在它的 test 集里：

**LANEIGE Water Bank Moisture Cream**（面霜）

| | 混合肌 | 干皮 | 中性肌 | 油皮 | 敏感肌 |
|---|---|---|---|---|---|
| 标签 | 0 | 1 | 1 | 0 | 1 |
| 预测 | 0.99 | 0.99 | 1.00 | 0.88 | 1.00 |

注意力最高：Polyisobutene (0.12)、Chenopodium Quinoa Seed Extract (0.11)、聚二甲基硅氧烷 (0.10)。注意力落在几个油脂/成膜成分上，但模型没有因此判定「不适合油皮」，混合肌和油皮都判错了。

**OLEHENRIKSEN Goodnight Glow Retin-ALT Sleeping Crème**（晚霜）

| | 混合肌 | 干皮 | 中性肌 | 油皮 | 敏感肌 |
|---|---|---|---|---|---|
| 标签 | 1 | 1 | 1 | 1 | 0 |
| 预测 | 0.92 | 1.00 | 0.99 | 0.67 | 0.03 |

5 个标签全部判对，包括少见的负标签「不适合敏感肌」。注意力最高：Dicaprylyl Carbonate (0.18)、Sodium Phytate (0.13)、Silybum Marianum Fruit Extract (0.09)，看不出和「敏感肌」有什么直接关系，所以这次判对不一定是因为模型理解了配方，注意力权重只能当作参考。

### 知识库特征（`--kb-feats`）

每个 token 额外查一个 15 维特征：刺激等级 / 3、致痘等级 / 5，以及知识库 13 个 tag（香精、致敏香料、维A类、AHA、BHA、高位酒精、防晒剂……）的 one-hot；经无偏置线性层投影到 d_model 后加到 token embedding 上，未识别成分是零向量。其余设置与上表完全相同（同种子、同划分、同超参），不加特征的模型与上表是同一批结果。

| 设置 | 方法 | Combination | Dry | Normal | Oily | Sensitive | macro-F1 | macro-AUC |
|---|---|---|---|---|---|---|---|---|
| 丢弃全 0 行 | SkinTypeNet | 0.979 | 0.935 | 0.976 | 0.905 | 0.796 | 0.918 ± 0.008 | 0.619 ± 0.038 |
| 丢弃全 0 行 | + 知识库特征 | 0.986 | 0.935 | 0.983 | 0.922 | 0.821 | 0.929 ± 0.005 | 0.591 ± 0.054 |
| 保留全 0 行 | SkinTypeNet | 0.776 | 0.743 | 0.763 | 0.725 | 0.684 | 0.738 ± 0.051 | 0.641 ± 0.040 |
| 保留全 0 行 | + 知识库特征 | 0.764 | 0.727 | 0.756 | 0.719 | 0.679 | 0.729 ± 0.031 | 0.635 ± 0.029 |

**没有可靠的提升。** 丢弃全 0 行时 macro-F1 +0.011，但 macro-AUC −0.028（5 个种子里 2 升 3 降），说明 F1 的上升主要是预测更偏向「适合」，而不是排序更准；保留全 0 行时两个指标都略降（AUC 3 升 2 降）。仍然不如词袋逻辑回归（AUC 0.679 / 0.702）。原因推测：

- 知识库成分本来就各有独立的 token embedding，这 15 维是 token id 的固定函数，对见过的成分不提供新信息，只能作为「相似成分共享方向」的先验；
- 只有 27.5% 的 token 被识别，13 个 tag 里多数只覆盖 1~3 个成分；
- 标签本身信号弱（见上），先验帮不上多少。

所以默认不开启。

### 产品类别分类（`--task category`）

用 CSV 的 `Label` 列（Cleanser / Eye cream / Face Mask / Moisturizer / Sun protect / Treatment）做 6 分类。这个任务不用肤质标签，全 0 行也保留，清洗和去重同上，共 1,213 条（971/121/121）。同成分表出现在多个类别时只保留第一条。SkinTypeNet 换成 softmax + 交叉熵（6 类 170~298 条，较均衡，不加权），d_model 固定为 128（与预训练模型一致，保证和 [Pretraining](#pretraining) 的对比公平），在种子 42 的验证集上从 lr ∈ {1e-3, 3e-4} 选了 3e-4，40 epoch；逻辑回归是多项 LR，`class_weight="balanced"`，C 按验证集 macro-F1 选。5 个种子平均：

| 方法 | Cleanser | Eye cream | Face Mask | Moisturizer | Sun protect | Treatment | accuracy | macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 全部预测为训练集最多的类 | — | — | — | — | — | — | 0.167 ± 0.016 | 0.048 ± 0.004 |
| 词袋 + 逻辑回归 | 0.642 | 0.478 | 0.511 | 0.394 | 0.644 | 0.416 | **0.509 ± 0.031** | **0.514 ± 0.035** |
| SkinTypeNet | 0.452 | 0.356 | 0.467 | 0.382 | 0.520 | 0.223 | 0.405 ± 0.041 | 0.400 ± 0.046 |
| SkinTypeNet + 知识库特征 | 0.526 | 0.359 | 0.503 | 0.343 | 0.666 | 0.226 | 0.438 ± 0.031 | 0.437 ± 0.030 |

- 成分表里**确实有类别信号**（两个模型都远高于平凡基线），这和肤质任务不同，说明前面肤质结果差主要是标签问题，而不只是模型问题。
- 但 SkinTypeNet 比逻辑回归低约 0.11。知识库特征在这个任务上 +0.037（5 个种子 3 升 2 降），主要来自防晒（防晒剂 tag）和洁面，但差值小于种子间标准差，不算可靠。1 千条左右的数据量对 2 层 Transformer 偏少，词袋线性模型更稳。
- 逻辑回归里防晒和洁面最好认，面霜和 Treatment（精华等）最难。推测前两者分别有防晒剂、表面活性剂这类标志性成分，后两者成分重叠较大（未逐一验证）。

复现：

```bash
python -m skinsight.baseline --csv data/cosmetics.csv --task category --seed 1
python -m skinsight.train --csv data/cosmetics.csv --task category --seed 1 --epochs 40 --lr 3e-4 --d-model 128
python -m skinsight.train --csv data/cosmetics.csv --seed 1 --epochs 40 --lr 1e-3 --d-model 128 --kb-feats
```

## Pretraining

丝芙兰数据只有 1 千条左右，于是试了在更大的**无标签**成分表语料上做自监督预训练，再微调到上面两个任务。

> **知识库版本**：本节的语料统计、预训练和微调结果都基于旧版知识库 **v0.1**，扩充到 v0.2 后没有重跑。

### 语料：Open Beauty Facts

预训练语料来自 [Open Beauty Facts](https://world.openbeautyfacts.org/)（英文站的 CSV 导出 `en.openbeautyfacts.org.products.csv.gz`），数据以 [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/) 授权，© Open Beauty Facts contributors。原始数据和处理后的语料都放在 `data/openbeautyfacts/` 下，**不入库**，需要自行下载。

| | 条数 |
|---|---|
| 导出中的产品 | 64,237 |
| 有成分表文本的 | 17,626 |
| − 清洗后不是成分表（同 `clean_ingredients`） | 7 |
| − 不足 5 个成分 | 2,289 |
| − 成分表完全相同的重复 | 686 |
| − 与 cosmetics.csv 某条成分表完全相同（防止下游 test 泄漏） | 5 |
| **语料** | **14,639**（预训练 train/val = 13,908/731） |

语料平均 20.6 个成分（丝芙兰 36.3），知识库识别率 26.7%；按出现次数计，丝芙兰成分表里 88.2% 的成分在语料中也出现过。

### 方法

- **任务**：masked ingredient modeling。每条成分表随机选 15% 的成分（pad/unk 不选，每条至少 1 个）作为目标，其中 80% 换成 `[MASK]`、10% 换成随机成分、10% 不变，用 encoder 输出预测原成分。
- **模型**：和下游完全同构的 SkinTypeNet encoder（token + 排序位置 embedding、2 层 Transformer，d_model=128），输出层与 token embedding 共享权重（logits 乘 d^-0.5，否则 N(0,1) 初始化的 embedding 让初始 loss 高达 17）。
- **词表**：预训练 train 部分出现 ≥ 2 次的成分 ∪ cosmetics.csv 成分表（**只用文本、不用任何标签**）中出现 ≥ 2 次的成分 ∪ 知识库，共 9,106 个；MLM 本身只在 OBF 语料上做，不看丝芙兰成分表。预训练 train 集 unk 比例 14.9%。
- **训练**：AdamW lr 1e-3，5% warmup + 线性衰减，batch 64，CPU 上限 55 分钟（39 个 epoch，55.2 分钟后停止）。

### 预训练指标

预训练 val 集（731 条，1,869 个被遮成分，固定遮罩种子）：

| | top-1 | top-5 |
|---|---|---|
| 频率基线（总猜训练集最常见的成分） | 0.044 | 0.154 |
| 预训练模型 | **0.303** | **0.455** |

模型确实学到了成分表的共现结构，远好于频率基线。

### 微调对比

下游划分、5 个种子、test 集全部与 [Results](#results) 相同；「从头训练」和逻辑回归就是上面表里的同一批结果。预训练模型的两种用法：

- **pre + 全量微调**：加载 encoder，所有参数都训练；lr 在种子 42 验证集上从 {1e-4, 3e-4} 选了 3e-4。
- **pre + 冻结 encoder**：只训练注意力池化和分类头；lr 从 {3e-4, 1e-3} 选了 1e-3（肤质任务 3e-4 最优 epoch 是第 1 轮，退化，不参与选择）。

肤质任务看 macro-AUC，类别任务看 macro-F1：

| 方法 | 肤质 AUC（丢弃全 0 行） | 肤质 AUC（保留全 0 行） | 类别 macro-F1 |
|---|---|---|---|
| 平凡基线 | 0.500 | 0.500 | 0.048 ± 0.004 |
| 词袋 + 逻辑回归 | **0.679 ± 0.061** | **0.702 ± 0.032** | **0.514 ± 0.035** |
| SkinTypeNet 从头训练 | 0.619 ± 0.038 | 0.641 ± 0.040 | 0.400 ± 0.046 |
| pre + 全量微调 | 0.586 ± 0.042 | 0.669 ± 0.034 | 0.468 ± 0.059 |
| pre + 冻结 encoder | 0.630 ± 0.057 | 0.633 ± 0.026 | 0.413 ± 0.075 |

（对应的 macro-F1：肤质丢弃全 0 行时 0.911 / 0.821，保留全 0 行时 0.746 / 0.711；类别 accuracy 0.468 / 0.425。冻结 encoder 在丢弃全 0 行时 F1 只有 0.821，是因为它预测出了更多「不适合」，在正样本占九成的数据上被 F1 惩罚。）

- **类别任务：预训练有帮助。** 全量微调比从头训练高 0.068，**5 个种子全部提升**；但仍比逻辑回归低 0.046。
- **肤质任务：没有可靠的结论。** 保留全 0 行时全量微调 +0.028（3 升 2 降），丢弃全 0 行时反而 −0.033（1 升 4 降）；冻结 encoder 在两个设置下和从头训练相差都在 0.01 左右。这些差值都小于种子间标准差（0.03~0.06）。肤质标签本身的信号太弱（见 [数据](#数据)），更好的成分表示也救不回来。
- **逻辑回归在三列里仍然都是最好的。** 1.4 万条无标签语料让 Transformer 在类别任务上追近了一半差距，但不足以超过词袋线性模型。

### 最近邻检查

对出现 ≥ 20 次的成分，在 token embedding 上找余弦最近邻：

- **直接用 embedding**：最近邻基本没有规律（例如 Glycerin → benzophenone-3、Propylene Glycol、microcrystalline wax）。原因是 embedding 是 N(0,1) 初始化、9 千维词表，训练只让它相对初始值移动了一小部分，余弦相似度主要还是初始噪声。
- **减去初始值（W − W0，再中心化）**：只看训练学到的部分，出现了一些合理的簇：
  - Parfum → perfume、parfum/fragrance 的其他写法、薰衣草油、甜橙果皮油、苯乙醇（香精和精油/香料成分）；
  - Zinc Oxide → Titanium Dioxide、talc、kaolin（矿物粉体/物理防晒）；
  - Dimethicone → cyclopentasiloxane、Carbomer、glycol stearate（硅油和质地成分）。
- **活性成分没有聚在一起**：Niacinamide、Retinol、Salicylic Acid 的最近邻都是不相关的成分（Retinol 在语料里只出现 37 次）。也就是说预训练学到的主要是「香精类 / 粉体类 / 基质类」这种配方结构，而不是功效，这和它对类别任务有帮助、对肤质任务帮助不大的结果是一致的。

### 复现

```bash
# 1. 下载 Open Beauty Facts 导出（约 18 MB），放到 data/openbeautyfacts/
# 2. 准备语料（去掉与 cosmetics.csv 完全相同的成分表）
python -m skinsight.pretrain_data --obf data/openbeautyfacts/en.openbeautyfacts.org.products.csv.gz \
    --exclude-csv data/cosmetics.csv --out data/openbeautyfacts/pretrain.jsonl
# 3. 预训练（CPU 约 55 分钟）
python -m skinsight.pretrain --corpus data/openbeautyfacts/pretrain.jsonl --cosmetics-csv data/cosmetics.csv \
    --epochs 40 --time-limit 55 --out checkpoints/pretrained.pt
# 4. 微调：全量 / 冻结 encoder（d_model 等结构参数取自预训练 checkpoint）
python -m skinsight.train --csv data/cosmetics.csv --seed 1 --epochs 40 --lr 3e-4 --init-ckpt checkpoints/pretrained.pt
python -m skinsight.train --csv data/cosmetics.csv --seed 1 --epochs 40 --lr 1e-3 --init-ckpt checkpoints/pretrained.pt --freeze-encoder
python -m skinsight.train --csv data/cosmetics.csv --task category --seed 1 --epochs 40 --lr 3e-4 --init-ckpt checkpoints/pretrained.pt
```

## Roadmap

- [x] OCR：拍照识别成分表（RapidOCR，本地运行）
- [ ] 根据肤质和所在城市的气候生成护肤流程
- [x] Web Demo（Gradio，可部署到魔搭创空间）
- [x] 扩充知识库（v0.2，377 个成分），中文名对接《已使用化妆品原料目录（2021年版）》

## 免责声明

本项目用于学习和信息参考，不构成医疗或皮肤科诊疗建议。致痘评分来自早期动物实验汇总，仅供参考；成品的实际表现取决于浓度和整体配方。皮肤出现持续不适请咨询医生。

## License

MIT

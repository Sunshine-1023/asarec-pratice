# FashionRec-Transformer 项目完整指南

> 本文档面向「第一次接触推荐系统」的读者，从零讲清楚：  
> **数据怎么来 → 模型怎么训 → 四路怎么召回 → 怎么融合 → 怎么评估 → 怎么搜权重**。  
> 读完后，你应该能独立跑通整条实验链路。

---

## 目录

1. [这个项目在做什么？](#1-这个项目在做什么)
2. [核心概念速查（小白必读）](#2-核心概念速查小白必读)
3. [项目目录结构](#3-项目目录结构)
4. [整体流程一张图](#4-整体流程一张图)
5. [第一阶段：数据准备](#5-第一阶段数据准备)
6. [第二阶段：训练 SASRecF](#6-第二阶段训练-sasrecf)
7. [第三阶段：四路召回](#7-第三阶段四路召回)
8. [第四阶段：多路融合](#8-第四阶段多路融合)
9. [第五阶段：离线评估（MAP@12）](#9-第五阶段离线评估map12)
10. [第六阶段：权重搜索（valid 调参）](#10-第六阶段权重搜索valid-调参)
11. [从头到尾：完整命令清单](#11-从头到尾完整命令清单)
12. [输出文件说明](#12-输出文件说明)
13. [常见问题 FAQ](#13-常见问题-faq)
14. [与 H&M Kaggle 竞赛的关系](#14-与-hm-kaggle-竞赛的关系)

---

## 1. 这个项目在做什么？

### 1.1 一句话总结

用 **H&M 时尚购买数据**，预测每个用户「下一周可能会买什么衣服」，最终给每个用户推荐 **12 个不重复的商品（article_id）**，并用 **MAP@12** 衡量推荐质量。

### 1.2 技术路线

本项目不是只靠一个深度学习模型，而是：

```
SASRecF（序列模型）
    +
Popular（全局热门）
    +
Category Popular（类别热门）
    +
Item2Item（商品共现）
    ↓
按用户活跃度加权融合
    ↓
Top-12 推荐列表
    ↓
MAP@12 离线评估
```

**为什么要多路召回？**

| 用户类型 | 特点 | 哪路更有用 |
|----------|------|------------|
| 老用户（买过很多） | 有丰富历史 | SASRecF、Item2Item |
| 新用户（买过 1~2 次） | 历史很少 | Popular、Category Popular |
| 冷启动（没买过） | 无历史 | Popular、Category Popular |

单一模型很难同时照顾好所有用户，所以用 **多路召回 + 融合** 是工业界常见做法。

---

## 2. 核心概念速查（小白必读）

| 术语 | 通俗解释 |
|------|----------|
| **user_id / customer_id** | 用户编号 |
| **item_id / article_id** | 商品编号（本项目统一规范为 10 位字符串，如 `0706016001`） |
| **交互 (interaction)** | 一次购买记录：某用户在某时间买了某商品 |
| **历史 (history)** | 评估时刻之前，用户已经买过的所有商品 |
| **标签 / Ground Truth** | 评估周内用户实际购买的商品（用来算 MAP） |
| **召回 (recall)** | 从全库商品中先粗筛出 Top-50 或 Top-100 候选 |
| **融合 (fusion)** | 把多路候选合并排序，选出最终 Top-12 |
| **MAP@12** | 主评估指标，衡量 Top-12 推荐有多准（越高越好） |
| **valid** | 验证集，用来调权重、选模型 |
| **test** | 测试集，只在权重定好后跑一次，报最终成绩 |
| **exclude_seen** | 是否从推荐中排除用户历史已购商品（默认不排除，允许复购） |

---

## 3. 项目目录结构

```
FashionRec-Transformer/
├── configs/
│   └── sasrecf.yaml              # SASRecF 训练配置
├── data/
│   ├── raw/                      # 原始 Kaggle 数据（不上传 Git）
│   │   ├── transactions_train.csv
│   │   ├── articles.csv
│   │   ├── customers.csv
│   │   └── filtered/             # 可选：采样后的缩小数据
│   └── processed/
│       ├── hm/                   # 【主评估用】时序切分数据
│       │   ├── hm.inter
│       │   ├── hm.train.inter    # 前 4 周
│       │   ├── hm.valid.inter    # 倒数第 2 周（标签）
│       │   └── hm.test.inter     # 最后 1 周（标签）
│       └── hm_seq/               # 【训练用】RecBole 序列格式
│           ├── hm_seq.train.inter
│           ├── hm_seq.valid.inter
│           ├── hm_seq.test.inter
│           └── hm_seq.item         # 商品类别特征（SASRecF 需要）
├── outputs/
│   ├── checkpoints/sasrecf/      # 训练好的模型权重
│   ├── recommendations/            # 召回 & 融合结果 CSV
│   └── evaluation/                 # 评估指标 JSON
├── src/
│   ├── data/                     # 数据预处理
│   ├── recall/                   # 各路召回逻辑
│   ├── fusion/                   # 加权融合
│   └── evaluate/                 # 离线评估 & 权重搜索
├── run_sasrecf.py                # 训练 SASRecF 入口
├── run_sasrecf_recall.py         # 导出 SASRecF 召回
├── run_weight_search.py          # valid 权重搜索入口
└── docs/
    └── PROJECT_GUIDE.md          # 本文档
```

---

## 4. 整体流程一张图

```
┌─────────────────────────────────────────────────────────────────┐
│                        原始 Kaggle 数据                          │
│              transactions_train.csv + articles.csv               │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  可选: src/data/filter.py    │  缩小数据，加快实验
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  src/data/preprocess.py      │  CSV → hm.inter
              │  src/data/split.py           │  4w train + 1w valid + 1w test
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
  data/processed/hm/                    run_sasrecf.py
  (主评估口径)                          ├─ 转 hm_seq 序列格式
         │                              ├─ build_item_features
         │                              └─ RecBole 训练 SASRecF
         │                                       │
         │                              outputs/checkpoints/sasrecf/
         │                                       │
         │                              run_sasrecf_recall.py
         │                              → sasrecf_valid.csv / test.csv
         │                                       │
         └───────────────────┬───────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  offline_eval / weight_search │
              │  在线计算:                    │
              │    popular                    │
              │    category_popular           │
              │    item2item                  │
              │  + 读取 sasrecf_*.csv         │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  weighted_fusion.py           │
              │  四路 rank 加权融合 → Top-12   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  MAP@12 离线评估              │
              │  valid 搜权重 → test 报分     │
              └─────────────────────────────┘
```

---

## 5. 第一阶段：数据准备

### 5.1 你需要准备什么？

从 [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) 下载数据，放到 `data/raw/`：

| 文件 | 是否必需 | 用途 |
|------|----------|------|
| `transactions_train.csv` | ✅ 必需 | 用户购买记录 |
| `articles.csv` | ✅ 必需（SASRecF） | 商品类别特征 |
| `customers.csv` | 可选 | 当前主线未使用 |

原始 CSV 长这样：

```csv
t_dat,customer_id,article_id,price,sales_channel_id
2018-09-20,1234567890,0706016001,0.05,2
...
```

### 5.2 可选：数据过滤（小规模快速实验）

**脚本：** `python -m src.data.filter`  
**代码：** `src/data/filter.py`

如果你电脑跑不动全量数据，可以先采样：

| 过滤规则 | 默认值 |
|----------|--------|
| 时间窗口 | 最近 6 周 |
| 保留热门商品 | Top 30,000 |
| 活跃用户 | 购买 ≥ 5 次 |
| 每用户最多行为 | 50 条 |

输出到 `data/raw/filtered/`，后续 `preprocess` 会**自动优先使用**过滤后的数据。

### 5.3 预处理：转成 RecBole 格式

**脚本：** `python -m src.data.preprocess`  
**代码：** `src/data/preprocess.py`

做了什么：

1. 读取 `transactions_train.csv`
2. 只保留最近 **6 周** 数据
3. 过滤掉购买次数 < 5 的用户
4. 转成 RecBole 标准三列格式

输出文件：`data/processed/hm/hm.inter`

```
user_id:token    item_id:token    timestamp:float
1234567890       0706016001       1537401600
...
```

### 5.4 时序切分：train / valid / test

**脚本：** `python -m src.data.split`  
**代码：** `src/data/split.py`

把 6 周数据按时间切成三段（**不能随机打乱，必须按时间**）：

```
|<──────────── 6 周 ────────────>|

|← 4 周 train →|← 1w valid →|← 1w test →|
```

| 文件 | 时间 | 角色 |
|------|------|------|
| `hm.train.inter` | 第 1~4 周 | 训练历史、建召回索引 |
| `hm.valid.inter` | 第 5 周 | **验证标签**（调权重用） |
| `hm.test.inter` | 第 6 周 | **测试标签**（最终报分） |

### 5.5 重要：两套数据并存

本项目有 **两条数据轨道**，初学者最容易搞混：

| 目录 | 用途 | 谁在用 |
|------|------|--------|
| `data/processed/hm/` | 竞赛式时序切分 | **离线评估主逻辑**（popular、融合、MAP@12） |
| `data/processed/hm_seq/` | RecBole 序列样本格式 | **SASRecF 训练 & 召回导出** |

**离线评估的核心思想：用过去预测未来**

| 评估阶段 | 用户历史从哪来 | 真实标签从哪来 |
|----------|----------------|----------------|
| valid | `hm.train.inter`（前 4 周） | `hm.valid.inter`（第 5 周买了什么） |
| test | `hm.train.inter` + `hm.valid.inter` | `hm.test.inter`（第 6 周买了什么） |

---

## 6. 第二阶段：训练 SASRecF

### 6.1 SASRecF 是什么？

- **SASRec**：用 Transformer 读用户购买序列，预测下一个商品
- **SASRecF**：在 SASRec 基础上，额外利用商品的 **类别特征**（颜色、部门、款式等）

本项目用 [RecBole](https://recbole.io/) 框架训练，配置文件：`configs/sasrecf.yaml`

### 6.2 训练入口

```bash
python run_sasrecf.py
```

等价于 `python run_sasrec.py --config configs/sasrecf.yaml`

**`run_sasrecf.py` 会自动帮你做这些事：**

```
1. preprocess + split        （若没加 --skip-preprocess）
2. hm.*.inter → hm_seq.*.inter  （转成序列样本格式）
3. build_item_features       （从 articles.csv 生成 hm_seq.item）
4. RecBole 训练 SASRecF      （早停指标 MAP@12）
5. 保存 checkpoint           → outputs/checkpoints/sasrecf/*.pth
```

### 6.3 hm → hm_seq 转换做了什么？

普通 `hm.train.inter` 每行只是一次购买：

```
用户A, 商品1, 时间1
用户A, 商品2, 时间2
用户A, 商品3, 时间3
```

`hm_seq.train.inter` 转成序列预测样本：

```
用户A, [商品1 商品2], 长度2, 目标商品3, 时间3
```

含义：**给定历史 [商品1, 商品2]，预测用户会买商品3**。

### 6.4 关键训练参数（configs/sasrecf.yaml）

| 参数 | 值 | 说明 |
|------|-----|------|
| `model` | SASRecF | 带特征的序列模型 |
| `MAX_ITEM_LIST_LENGTH` | 100 | 最多看最近 100 次购买 |
| `hidden_size` | 128 | 模型维度 |
| `loss_type` | CE | 交叉熵（全库 softmax） |
| `valid_metric` | MAP@12 | 早停看 MAP@12 |
| `recall_top_k` | 100 | 召回先取 Top-100 |
| `checkpoint_dir` | outputs/checkpoints/sasrecf | 模型保存位置 |

### 6.5 数据已准备好时跳过预处理

```bash
python run_sasrecf.py --skip-preprocess
```

---

## 7. 第三阶段：四路召回

召回 = **从几十万商品里，先粗筛出几十个候选**，后面融合再精选 Top-12。

### 7.1 通道总览

| 通道 | 代码文件 | 核心思路 | 默认 Top-K |
|------|----------|----------|------------|
| **SASRecF** | `src/recall/sasrec_recall.py` | 深度学习序列预测 | 100 |
| **Popular** | `src/recall/popular.py` | 全局时间衰减热门 | 50 |
| **Category Popular** | `src/recall/category_popular.py` | 用户买过的类别里的热门 | 50 |
| **Item2Item** | `src/recall/item2item.py` | 「买了 A 的人也买了 B」 | 50 |

### 7.2 通道 1：SASRecF（需要单独导出）

**脚本：**

```bash
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test
```

**输出：**

- `outputs/recommendations/sasrecf_valid.csv`
- `outputs/recommendations/sasrecf_test.csv`

**CSV 格式：**

```csv
user_id,item_id,score,rank,channel
1234567890,0706016001,0.85,1,sasrecf
1234567890,0123456789,0.72,2,sasrecf
...
```

**逻辑简述：**

1. 加载训练好的 checkpoint
2. 对每个评估用户，取其历史序列
3. 模型对全库商品打分，取 Top-100
4. 写入 CSV

### 7.3 通道 2：Popular（全局热门）

**代码：** `src/recall/popular.py`  
**无需单独跑脚本**，在 `offline_eval` 时自动计算。

**逻辑：**

用时间衰减公式统计商品热度：

```
hot_score = 0.5×最近1周热度 + 0.3×最近2周 + 0.15×最近4周 + 0.05×最近8周
```

对所有用户返回相同的热门列表（个性化程度低，但对冷启动用户有用）。

### 7.4 通道 3：Category Popular（类别热门）

**代码：** `src/recall/category_popular.py`

**逻辑（分 4 步）：**

```
1. 看用户最近 10 次购买
2. 推断用户偏好哪些类别（如：女装 / 裤子 / 黑色）
3. 在这些类别桶里找热门商品
4. 合并打分，返回 Top-50
```

适合「用户买过几条裤子，再推荐同类热门」的场景。

### 7.5 通道 4：Item2Item（商品共现）

**代码：** `src/recall/item2item.py`

**逻辑：**

```
1. 统计最近 8 周内：同一用户买了 A 又买了 B 的次数
2. 得到 A → B 的共现分数
3. 用用户最近 10 个购买作种子
4. 聚合邻居分数，返回 Top-50
```

适合老用户：「你买过 A，推荐经常和 A 一起买的 B」。

### 7.6 四路召回对比

| | SASRecF | Popular | Category Popular | Item2Item |
|--|---------|---------|------------------|-----------|
| 需要训练 | ✅ | ❌ | ❌ | ❌ |
| 需要历史 | 越多越好 | 不需要 | 少量即可 | 需要一些 |
| 个性化程度 | 高 | 低 | 中 | 中高 |
| 冷启动表现 | 差 | 好 | 好 | 差 |

---

## 8. 第四阶段：多路融合

### 8.1 融合代码

**文件：** `src/fusion/weighted_fusion.py`  
**核心函数：** `fuse_candidates()`

### 8.2 融合公式

对每一路召回的候选列表，按排名给分：

```
rank_score = 1 / (rank + 1)

rank=0 → 1.0
rank=1 → 0.5
rank=2 → 0.333
...
```

同一商品如果在多路都出现，分数**累加**：

```
final_score[item] = Σ (通道权重 × rank_score)
```

最后按 `final_score` 降序排列，取 **Top-12**。

**保证：每个用户的 Top-12 不会有重复 article_id。**

### 8.3 按用户活跃度给不同权重

用户按历史购买次数分 4 档：

| 分层 | 条件 | 含义 |
|------|------|------|
| `high` | 购买 ≥ 10 次 | 高活跃用户 |
| `medium` | 购买 3~9 次 | 中活跃用户 |
| `low` | 购买 1~2 次 | 低活跃用户 |
| `cold_start` | 购买 0 次 | 冷启动用户 |

默认权重模板（`ACTIVITY_WEIGHTS`）：

| 分层 | SASRecF | Popular | Category Popular | Item2Item |
|------|---------|---------|------------------|-----------|
| high | 0.60 | 0.10 | 0.10 | 0.20 |
| medium | 0.40 | 0.15 | 0.15 | 0.30 |
| low | 0.15 | 0.35 | 0.25 | 0.25 |
| cold_start | 0.00 | 0.55 | 0.30 | 0.15 |

直觉：历史越多，越信任 SASRecF；历史越少，越依赖热门类召回。

### 8.4 exclude_seen：是否排除已购商品

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `exclude_seen=False`（**默认**） | 允许推荐用户买过的商品 | H&M 服装复购很常见 |
| `exclude_seen=True` | 过滤掉历史中出现过的商品 | 只想推新商品时 |

权重搜索会**两种都试**，选 valid MAP@12 更高的那种。

---

## 9. 第五阶段：离线评估（MAP@12）

### 9.1 评估入口

```bash
# 验证集（调参 / 看效果）
python -m src.evaluate.offline_eval --eval-split valid

# 测试集（用搜索到的最佳权重）
python -m src.evaluate.offline_eval \
  --eval-split test \
  --weights-json outputs/evaluation/best_fusion_weights.json
```

**代码：** `src/evaluate/offline_eval.py`

### 9.2 评估流程（每个用户）

```
1. 读取用户历史（train 或 train+valid）
2. 四路召回 → 得到候选
3. 按活跃度选权重 → 融合 → Top-12 预测列表
4. 读取 valid/test 周真实购买 → actual 集合
5. 计算该用户的 AP@12
6. 所有用户取平均 → MAP@12
```

### 9.3 MAP@12 怎么算？（小白版）

假设某用户第 5 周实际买了 `{A, B}` 两件商品。  
模型推荐的 Top-12 中，第 2 位是 A，第 5 位是 B：

```
位置 2 命中 A → 精确率 = 1/2 = 0.5
位置 5 命中 B → 精确率 = 2/5 = 0.4

AP@12 = (0.5 + 0.4) / min(2, 12) = 0.9 / 2 = 0.45
```

所有用户的 AP@12 取平均 = **MAP@12**。

### 9.4 评估设计要点

| 要点 | 本项目做法 |
|------|------------|
| 真实标签去重 | `actual` 用 `set`，同一商品买多次只算 1 个 relevant |
| 推荐列表去重 | 融合输出保证每用户 Top-12 不重复 |
| 评估用户范围 | 只评估 valid/test 文件中**有购买行为**的用户 |
| 默认允许复购 | `exclude_seen=False` |
| rank 公式统一 | 所有通道融合时统一 `1/(rank+1)`，rank 从 0 开始 |

### 9.5 同时输出的其他指标

| 指标 | 含义 |
|------|------|
| Recall@12 | 真实商品中被推荐命中的比例 |
| NDCG@12 | 考虑排名位置的折扣命中 |
| Hit@12 | Top-12 里有没有至少命中 1 个 |

**主指标仍是 MAP@12。**

---

## 10. 第六阶段：权重搜索（valid 调参）

### 10.1 为什么要搜权重？

手调 `ACTIVITY_WEIGHTS` 很费时。权重搜索在 **valid 集**上自动尝试不同权重组合，找 MAP@12 最高的配置。

### 10.2 入口

```bash
python run_weight_search.py
# 或
python -m src.evaluate.weight_search --eval-split valid
```

**代码：** `src/evaluate/weight_search.py`

### 10.3 搜索策略

1. **只在 valid 上搜**（不能用 test 调参）
2. 四路召回只算一次，搜索时只改融合权重（快）
3. 粗粒度网格，步长 0.05
4. 坐标下降：逐层（high → medium → low → cold_start）优化
5. 分别搜索 `exclude_seen=False` 和 `True`，取更优者

### 10.4 输出

`outputs/evaluation/best_fusion_weights.json`

```json
{
  "protocol": "hm_fusion_weight_search",
  "eval_split": "valid",
  "exclude_seen": false,
  "best_map@12": 0.028,
  "best_weights": {
    "high":    {"sequence": 0.55, "popular": 0.15, "category_popular": 0.15, "item2item": 0.15},
    "medium":  {"sequence": 0.40, "popular": 0.20, "category_popular": 0.20, "item2item": 0.20},
    "low":     {"sequence": 0.10, "popular": 0.50, "category_popular": 0.25, "item2item": 0.15},
    "cold_start": {"sequence": 0.0, "popular": 0.70, "category_popular": 0.30, "item2item": 0.0}
  }
}
```

> 注意：JSON 里 `sequence` 就是 SASRecF 通道权重。

---

## 11. 从头到尾：完整命令清单

### 11.1 环境安装

```bash
conda activate ai          # 你的 Python 环境
pip install -r requirements.txt
```

### 11.2 标准全流程（推荐顺序）

```bash
# ── 第 1 步：数据准备 ──
# 可选：小规模采样
python -m src.data.filter

# 必需：预处理 + 切分
python -m src.data.preprocess
python -m src.data.split

# ── 第 2 步：训练 SASRecF ──
python run_sasrecf.py --skip-preprocess

# ── 第 3 步：导出 SASRecF 召回 ──
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test

# ── 第 4 步：valid 权重搜索 ──
python run_weight_search.py

# ── 第 5 步：离线评估 ──
# valid：看调参效果
python -m src.evaluate.offline_eval --eval-split valid

# test：最终成绩（用搜索到的权重）
python -m src.evaluate.offline_eval \
  --eval-split test \
  --weights-json outputs/evaluation/best_fusion_weights.json
```

### 11.3 一键从头跑（含数据预处理）

若数据还没处理过，训练脚本会自动 preprocess + split：

```bash
python run_sasrecf.py
python run_sasrecf_recall.py --eval-split valid
python run_sasrecf_recall.py --eval-split test
python run_weight_search.py
python -m src.evaluate.offline_eval --eval-split test \
  --weights-json outputs/evaluation/best_fusion_weights.json
```

### 11.4 各步骤对应文件速查

| 步骤 | 运行命令 | 核心代码 |
|------|----------|----------|
| 过滤 | `python -m src.data.filter` | `src/data/filter.py` |
| 预处理 | `python -m src.data.preprocess` | `src/data/preprocess.py` |
| 切分 | `python -m src.data.split` | `src/data/split.py` |
| 商品特征 | （训练时自动） | `src/data/build_item_features.py` |
| 训练 | `python run_sasrecf.py` | `run_sasrec.py` + `configs/sasrecf.yaml` |
| SASRecF 召回 | `python run_sasrecf_recall.py` | `src/recall/sasrec_recall.py` |
| 权重搜索 | `python run_weight_search.py` | `src/evaluate/weight_search.py` |
| 融合评估 | `python -m src.evaluate.offline_eval` | `src/evaluate/offline_eval.py` |
| 融合逻辑 | （评估时自动） | `src/fusion/weighted_fusion.py` |

---

## 12. 输出文件说明

### 12.1 数据文件

| 路径 | 内容 |
|------|------|
| `data/processed/hm/hm.train.inter` | 训练期购买记录 |
| `data/processed/hm/hm.valid.inter` | 验证期购买记录（标签） |
| `data/processed/hm/hm.test.inter` | 测试期购买记录（标签） |
| `data/processed/hm_seq/hm_seq.*.inter` | RecBole 序列训练数据 |
| `data/processed/hm_seq/hm_seq.item` | 商品类别特征 |

### 12.2 模型 & 召回

| 路径 | 内容 |
|------|------|
| `outputs/checkpoints/sasrecf/*.pth` | SASRecF 模型权重 |
| `outputs/recommendations/sasrecf_valid.csv` | SASRecF valid 召回 |
| `outputs/recommendations/sasrecf_test.csv` | SASRecF test 召回 |

### 12.3 融合 & 评估

| 路径 | 内容 |
|------|------|
| `outputs/recommendations/fusion_valid.csv` | valid 融合 Top-12 |
| `outputs/recommendations/fusion_test.csv` | test 融合 Top-12 |
| `outputs/evaluation/fusion_valid_metrics.json` | valid 指标 |
| `outputs/evaluation/fusion_test_metrics.json` | test 指标 |
| `outputs/evaluation/best_fusion_weights.json` | 搜索到的最佳权重 |

### 12.4 fusion_*.csv 格式

```csv
user_id,item_id,score,rank,split,channel
1234567890,0706016001,0.82,1,valid,fusion
1234567890,0123456789,0.65,2,valid,fusion
...
```

每个用户恰好 12 行（Top-12），`item_id` 不重复。

---

## 13. 常见问题 FAQ

### Q1：`hm` 和 `hm_seq` 有什么区别？

- `hm/`：原始时序切分，**离线评估**用这个
- `hm_seq/`：加了 `item_id_list` 序列列，**RecBole 训练**用这个

### Q2：为什么 SASRecF 召回要单独导出 CSV？

因为 SASRecF 推理需要 GPU + RecBole 模型，比较慢。导出一次后，权重搜索可以反复改融合权重而不用重新推理。

### Q3：Popular / Category Popular / Item2Item 要不要单独跑？

**不用。** 它们在 `offline_eval` 和 `weight_search` 里实时计算。

### Q4：valid 和 test 有什么区别？

| | valid | test |
|--|-------|------|
| 用途 | 调权重、选模型 | 最终报分 |
| 历史 | train（4 周） | train + valid（5 周） |
| 标签 | 第 5 周购买 | 第 6 周购买 |
| 可以反复跑吗 | ✅ 可以 | ⚠️ 权重定好后只跑一次 |

### Q5：MAP@12 多少算好？

H&M 竞赛全量数据上 MAP@12 通常在 **0.02~0.04** 量级（取决于数据规模和模型）。本项目目标是尽可能接近或超过 **0.03**。

### Q6：`run_recbole_*` 脚本是什么？

RecBole 原生评估的平行实验线，和主线 `offline_eval` 评估口径不同。**日常实验以 `offline_eval` 为准。**

### Q7：报错找不到 `sasrecf_valid.csv` 怎么办？

先跑召回导出：

```bash
python run_sasrecf_recall.py --eval-split valid
```

### Q8：报错找不到 `hm.train.inter` 怎么办？

先跑数据准备：

```bash
python -m src.data.preprocess
python -m src.data.split
```

---

## 14. 与 H&M Kaggle 竞赛的关系

| 竞赛要求 | 本项目 |
|----------|--------|
| 每用户推荐 12 个商品 | ✅ `final_top_k=12` |
| 推荐列表不重复 | ✅ 融合层 dict 去重 |
| 评估指标 MAP@12 | ✅ 主线指标 |
| 允许复购推荐 | ✅ 默认 `exclude_seen=False` |
| 全量提交用户 | 离线评估只评有购买的用户；正式提交需对全量客户生成推荐 |

> 正式 Kaggle 提交还需要额外的「全量用户推荐生成 + submission.csv 格式化」步骤，当前仓库聚焦离线评估链路。

---

## 附录：关键代码文件索引

| 文件 | 职责 |
|------|------|
| `src/data/filter.py` | 原始数据采样过滤 |
| `src/data/preprocess.py` | 交易 CSV → hm.inter |
| `src/data/split.py` | 时序切分 train/valid/test |
| `src/data/build_item_features.py` | articles.csv → hm_seq.item |
| `run_sasrec.py` | RecBole 训练主逻辑 |
| `run_sasrecf.py` | SASRecF 训练入口 |
| `src/recall/sasrec_recall.py` | SASRecF Top-K 导出 |
| `src/recall/popular.py` | 全局热门召回 |
| `src/recall/category_popular.py` | 类别热门召回 |
| `src/recall/item2item.py` | 商品共现召回 |
| `src/fusion/weighted_fusion.py` | 多路加权 rank 融合 |
| `src/evaluate/offline_eval.py` | 融合 + MAP@12 评估 |
| `src/evaluate/weight_search.py` | valid 权重网格搜索 |
| `configs/sasrecf.yaml` | SASRecF 超参数配置 |

---

*文档版本：2026-07，对应项目主线 SASRecF + 四路召回融合流程。*

# FashionRec 架构说明

## 设计目标

当前代码按“数据契约优先、候选先物化、排序可替换、产物按运行隔离”组织。重构不改变 SASRecF、Popular、Category Popular、Item2Item 和加权 RRF 的算法含义，主要解决 ID 不一致、脚本互相导入、候选重复计算、全局产物串用以及评估静默回退问题。

## 分层与依赖方向

```text
Makefile / fashionrec CLI（唯一公开入口）
        │
        ▼
fashionrec.pipeline + fashionrec.experiment（编排、配置、运行目录）
        │
        ├── fashionrec.data（预处理、时间切分、序列样本）
        ├── fashionrec.recall（通道接口、索引、候选生成）
        ├── fashionrec.candidates（候选并集与去重）
        ├── fashionrec.ranking（Weighted RRF / LightGBM 特征边界）
        └── fashionrec.evaluation（纯指标、权重搜索、报告）
                │
                ▼
          fashionrec.domain（ID 与 Candidate 契约）
```

所有实现位于标准 `src/fashionrec/` 包中。CLI 延迟加载领域命令，底层模块不导入 CLI 或 pipeline；流水线内部的子进程也只调用 `python -m fashionrec <command>`，不依赖具体实现文件路径。

## 核心数据契约

- 商品 ID：内部统一为十位字符串，例如 `706016001` 与 `0706016001` 都映射为 `0706016001`。
- 候选记录：固定包含 `user_id`、`item_id`、`channel`、`score`、`rank`、`split`。
- 候选并集：同一用户、商品、通道仅保留一条证据；同一商品来自多个通道时保留多路证据，供融合或学习排序使用。
- 时间因果：切分前不按未来总活跃度筛用户或截断历史；可选 Top-item 集合只在 train 拟合。`hm.model_train.inter` 仅负责模型拟合资格，完整 `hm.train.inter` 仍供 valid 历史与冷启动评估使用。
- 输入确定性：未启用 `--with-filter` 时始终读取原始 transactions；启用后只读取本次刚生成的 filtered transactions，不根据历史文件是否存在自动切换。
- checkpoint 口径：每个 RecBole 验证 epoch 都进入粗筛，按 `valid_metric_bigger` 方向保留真正的指标 Top-5；最终模型只按完整 valid 用户周的 MAP@12 选择，不读取 test 标签。
- checkpoint 隔离：shortlist 目标目录必须为空；相同 run/seed 重跑时直接失败并保留旧文件，复用旧结果应跳过训练，新训练应使用新的 run ID。

## 正式流水线

```text
数据准备
  → SASRecF 训练
  → valid 用户周 MAP@12 选择 checkpoint
  → SASRecF valid/test 召回
  → 规则召回 + 四路候选物化
  → valid 权重搜索
  → Weighted RRF valid/test 排序
  → 离线指标与运行清单
```

`fashionrec pipeline` 首先解析一次 `configs/experiment.yaml`，创建 `RunContext`，然后把同一份参数和同一个运行目录传给全部阶段。`max_user_history` 同时约束候选生成、用户分层和已购过滤；valid 与 test 排序同时读取 valid 搜索得到的同一权重文件。正式模式默认 `strict=True`：候选文件、序列召回等关键依赖缺失时直接失败，不再静默换模型或使用空通道。

单次运行目录：

```text
outputs/runs/<run_id>/
├── resolved_config.json
├── manifest.json
├── checkpoints/sasrecf/
├── checkpoints/sasrecf_selected.pth
├── recall/
├── candidates/
│   ├── valid.csv
│   └── test.csv
├── ranking/
├── evaluation/
│   └── sasrecf_checkpoint_selection.json
└── logs/
```

## 排序层

当前等价基线是 `WeightedRRFRanker`。它实现通用 `Ranker` 接口，离线评估不再直接拥有融合算法。

`src/fashionrec/ranking/features.py` 将候选并集透视为“一用户一商品”表，输出用户历史长度、通道覆盖数、最佳通道排名、最大通道分、各通道 `present / score / rank`、训练标签，以及 LightGBM LambdaRank 所需的用户 group sizes。

因此后续接入 `lightgbm.LGBMRanker(objective="lambdarank")` 时，只需新增模型训练与预测适配器，不需要改召回、候选或指标模块。

## 公开入口

正式实验统一使用 Makefile：

```bash
make pipeline WITH_FILTER=1
```

`Makefile` 只把目标和变量翻译为 `fashionrec pipeline` 参数，不复制阶段顺序或产物路径。分阶段运行必须提供同一个 `RUN_ID`，例如 `make train RUN_ID=exp-001` 和 `make downstream RUN_ID=exp-001`。调试旧产物时可设置 `STRICT=0`；为了避免不同实验串用文件，正式结果不要从 `outputs/recommendations` 或 `outputs/evaluation` 手工拼接。

Python 对外只提供 `fashionrec` 一个 console script 和 `python -m fashionrec` 一个模块入口。命令实现仍按 data、training、recall、ranking、evaluation 与 pipeline 分层存放，但不再暴露根目录启动脚本。

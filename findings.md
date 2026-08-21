# Findings

## 2026-08-20 当前阻断

- 默认 `ranking.enabled: false` 的 run-scoped RRF 主链可完整规划和执行。
- `ranking/dataset.py`、`train.py`、`predict.py` 与 CLI 已存在，但 pipeline 直接读取 `ranking/{train,valid,test}.parquet`，没有生成步骤。
- 当前 candidates 只支持 valid/test；训练快照候选尚未物化。
- 默认 data 步骤不传 `--build-labels/--build-user-features/--build-cross-features`，因此新协议输入不会生成。
- offline evaluation 与 weight search 仍读 `hm.valid.inter` / `hm.test.inter`。标签会聚合为 set，因此数量重复不会重复计 MAP，但 next-basket 辅助标签和 snapshot 口径未成为正式输入。
- pipeline 未透传 `repurchase/style/content_top_k`，YAML 200 实际退回模块默认 50。
- RRF 权重只覆盖 popular/category_popular/item2item/sequence；repurchase/style/content 权重为 0，只影响 union 截断。
- repurchase recall 把完整购买历史误当“当前 basket”排除，导致真正复购 SKU 基本被清空。
- `requirements.txt` 未声明 LightGBM；配置允许的 library/objective 与训练硬编码存在漂移。
- 现有全量单测通过（215 tests，exit 0）。

## 正确且需要保留的行为

- SASRecF 序列按购物日推进，同日商品共享历史，user-date-item 去重。
- valid 历史只含 train；test 历史含 train+valid；test 不用于调参。
- 正式 pipeline 的数据、checkpoint、召回、候选、排序、评估产物均按 run 隔离。

## 实现设计发现

- `build_cross_features(labels_dir=...)` 只会给正例对生成交叉特征；正确的排序表必须对全部候选对生成，因此交叉特征应在候选与 snapshot_date 对齐之后构建。
- popular/category/item2item/repurchase/style/content 的底层索引均已有或可透传 `as_of`，可据此为 train weekly snapshots 构造因果规则候选。
- 不能用在完整 train 上拟合后的 SASRecF 给早期 train snapshot 生成训练特征；首版训练快照应使用因果规则候选，valid/test 仍可加入正式 SASRecF 通道。
- `RankingDataset` 已支持 user/customer/item/cross 四类 join，新增命令主要负责：读取快照、生成/读取候选、构建全部候选交叉特征、加载标签和写 parquet。

## 2026-08-20 双链路隔离审计

- 当前只有 `ranking.enabled` 条件分支，不是两个显式 profile；使用同一配置文件切换容易误跑。
- `RunContext.initialize()` 会无条件重写 `resolved_config.json`，相同 run-id 配置漂移时没有保护。
- Makefile 分阶段目标没有 `--skip-ranker`；ranking 开启时 `make data/train/...` 会意外规划 ranking dataset/train。
- `ranker_active` 同时控制工业数据准备和 ranker 执行，导致无法只运行 industrial data 阶段；应拆成 `industrial_protocol` 与 `run_ranker_steps`。
- optimized 编排尚未把 `data/labels` 传给 weights/evaluate，因此训练与最终指标标签口径可能不一致。
- baseline 候选需显式限制为 `popular,category_popular,item2item`，再加 SASRecF 才是原四路；industrial 才启用 repurchase/style/content。
- 目标目录采用 `outputs/runs/<profile>/<run_id>/`，比只要求不同 run-id 更能防止两条链误覆盖。

## 已实现的隔离契约

- `make baseline` 固定 `configs/experiment_baseline.yaml` 与 baseline profile；只构建 Popular / Category Popular / Item2Item 三路规则索引，加 SASRecF 为原四路候选。
- `make industrial` 固定 `configs/experiment_industrial.yaml` 与 industrial profile；生成 events/baskets/labels/PIT user features，扩展候选并运行 ranking dataset/train/predict。
- `make ranker` 无条件绑定 industrial；通用分阶段目标显式 `--skip-ranker`，但 industrial 的数据准备和 next-basket 评估协议不会被关闭。
- weights/evaluate 在 industrial 下统一接收 run 内 `data/labels`；baseline 不接收该参数，保持旧评估口径。
- run 目录按 profile 分层；run-id 禁止路径分隔符；同 profile/run-id 的配置哈希变化会拒绝续跑。
- baseline 注册表按需构建索引，不再为未启用的 repurchase/style/content 支付构建成本。

## 2026-08-20 结构重构审查

- 技术组件分层整体合理；不应把 data/recall/ranking 整套复制进两个链路目录。
- `pipeline/orchestrator.py` 321 行，同时拥有两个 profile 的条件分支，是当前最主要的链路耦合点。
- `data/command.py` 527 行和 `evaluation/offline_eval.py` 693 行偏大，但本轮优先拆 workflow；它们的内部服务化可作为后续独立重构，避免一次迁移过大。
- `ranking/command.py` 373 行只服务 industrial 的因果 ranking table，应归属 `pipeline/industrial/ranker_dataset.py`；ranking 的 dataset/train/predict 算法模块继续共享。
- 配置目前有三份 experiment YAML；应改为 `configs/baseline/experiment.yaml` 与 `configs/industrial/experiment.yaml` 两个唯一 profile 来源。
- `configs/experiment.yaml` 与 baseline 仅实验名和注释不同，没有独立协议价值；无参数加载应直接默认 baseline 配置，删除第三副本。
- 为兼容仓库内旧分析/报告调用，最终保留 `configs/experiment.yaml`，但正式默认加载、Makefile 和文档示例统一指向 `configs/baseline/experiment.yaml`；它不再是第三条正式 workflow。
- 结构边界已由测试固化：baseline 不导入 industrial；共享组件不反向导入 pipeline；common stages 不读取 `ranking.enabled`；baseline DAG 不含 ranker 命令。
- 测试目录平铺约 40 个文件；本轮先迁移 pipeline、run context、Make/CLI 和 industrial materialization 测试，算法单测保持原位以控制变更面。

## 2026-08-20 应用级彻底隔离目标

- 当前仍由 `pipeline/common/stages.py` 同时构造两链 data/train/recall/candidates/weights/evaluate 命令；这属于运行复用，不满足应用所有权完全分开。
- 当前两链子步骤都调用 `python -m fashionrec <command>`，因此 data、训练、候选和评估仍经过同一公开命令入口。
- 下一步应让正式 DAG 分别调用 `python -m fashionrec.baseline <stage>` 和 `python -m fashionrec.industrial <stage>`；兼容 `fashionrec pipeline --profile` 可保留但不再是 Make 正式入口。
- 完全复制算法实现会制造修复漂移；隔离对象应是应用协议和 wrapper，底层稳定算法以 shared kernel 或 algorithm library 形式复用。
- `training.command.main` 与 `sasrec_recall.main` 已支持传入默认配置，适合由应用 wrapper 固定模型配置；checkpoint 命令需要 wrapper 强制 `--config`。
- 候选物化 service 已支持显式 `--channels`，因此 baseline/industrial 可以分别拥有不可变 registry，而无需复制索引与生成算法。
- weight/evaluate service 的 `labels_dir` 与 `ranker_scored_csv` 是当前协议切换点；应用 wrapper 可分别禁止或要求这些参数。
- 正式 DAG 现在分别调用 `python -m fashionrec.baseline <stage>` 与 `python -m fashionrec.industrial <stage>`；旧统一 CLI 仍可作为兼容入口，但 Makefile 和应用内部不再依赖它。
- 两套 `configs/*/models/sasrecf.yaml` 初始内容相同但路径和所有权独立，后续任一应用调参不需要修改另一套配置。
- 最终正式依赖关系为 `baseline application -> shared/algorithm libraries` 与 `industrial application -> shared/algorithm libraries`；应用之间无 import，应用也不再依赖旧 `fashionrec.pipeline` 包。
- `pipeline/`、`domain/`、`evaluation/metrics.py` 的旧路径仅保留薄 facade，避免现有脚本一次性失效；正式 Make/DAG 不经过这些兼容 workflow 入口。

## 2026-08-20 物理目录归属迁移

- 用户确认要进一步迁移成 `shared/{domain,interfaces,io,metrics}`、`baseline/{data,models,recall,ranking,evaluation,pipeline}`、`industrial/{data,models,recall,ranking,evaluation,pipeline}`。
- 当前 `baseline` / `industrial` 主要还是协议 wrapper，正式实现仍集中在顶层 `data/recall/ranking/training/evaluation`；因此目录外观接近目标，但物理所有权尚未达成。
- 工作区包含上一轮已验证但尚未提交的隔离改动；本轮必须在其上增量迁移，不能回滚或覆盖。
- 兼容策略：旧顶层 import 路径保留薄 facade；应用内部与正式 DAG 必须只引用自身模块或 `shared`，禁止继续引用旧实现目录。
- 迁移采用“wrapper 保持稳定、实现下沉到应用内部”的方式：例如 `baseline/data/command.py` 继续负责协议校验，但调用本应用的 `data/service.py`，从而保留现有 CLI 和 monkeypatch 测试边界。
- SASRecF 的训练、checkpoint 选择和 recall 实现分别复制进两套 `models/sasrecf`；LambdaRank 的模型训练/推理归入 `industrial/models/lambdarank`。
- Baseline 只拥有 Popular、CategoryPopular、Item2Item 和 SASRecF 所需实现；Industrial 拥有完整购物篮/PIT 数据与六路规则召回实现。
- 最终正式应用源码不再 import `fashionrec.data/recall/ranking/training/evaluation/candidates`；这些旧目录只剩 13 个 facade 文件，真实实现均位于两应用目录。
- 旧 import 使用模块对象别名而非重复加载源文件，保证 dataclass 类型身份、monkeypatch 和历史测试语义不变。
- 为避免导入 Recall 包时无条件加载 Torch，SASRecF 兼容导出保持延迟加载；这也消除了与 LightGBM/OpenMP 同进程测试时出现的原生崩溃。

## 2026-08-20 双链路完整性复审

- 本轮只读审计，不修改算法实现；“完整”拆成 DAG 可执行闭环、数据/标签语义一致、防泄漏、指标公平和真实全量运行验证五个层次。
- 上一轮已证明 251 项单测、CLI smoke、10/14 步 DAG 规划和产物命名空间通过，但尚未证明 3100 万行 H&M 数据上的整链训练成功或指标提升。
- DAG 产物路径目前闭环：两链均为 data → SASRecF train → valid checkpoint → valid/test recall → candidates → valid 权重 → valid/test evaluate；Industrial 额外在 candidates 后构建 ranking parquet、训练/预测 LambdaRank。
- Industrial 的 weight search 与最终 evaluate 都显式读取同一 run 的 `data/labels`；Baseline 明确不读取 next-basket labels。
- 配置里 Baseline 与 Industrial 都使用 26 周历史 + 1 周 valid + 1 周 test；两者主要差异在 ranking 开关与应用协议，不是时间窗口。
- 两链的 SASRecF 配置当前完全相同；Industrial 的序列通道仍是 Baseline SASRecF 的独立副本，不是不同的序列模型。
- SASRecF checkpoint 只用 valid 粗筛/精筛，不读取 test；但 valid 同时承担 RecBole 早停、checkpoint 精筛和 RRF 权重搜索，因此 valid 指标是调参内指标，最终泛化判断只能看 test。
- Baseline 的序列数据已固定按购物日去重和共享历史，`deduplicate_user_day_item` 配置目前只是记录到 manifest，并不是可切换开关；因此 Baseline 不再复刻旧交易行伪序列。
- `model_train` 只用 train 统计购买次数 ≥5 的用户训练模型，完整 train 仍用于 valid 历史；这避免了用 valid/test 活跃度筛训练用户。

## 2026-08-20 双链路完整性复审结论

- 结论分层：两套单窗口 DAG 和产物依赖闭环完整，应用/profile/运行目录隔离成立；Baseline 逻辑可作为稳定对照；Industrial 的 PIT 截断和 next-basket 标签方向正确，但当前不能认定为全量可运行、语义完全一致的工业链。
- P1：LambdaRank 的 train 候选只来自六路规则召回，valid/test 候选额外包含 SASRecF。训练表仍声明 `sasrecf_present/score/rank`，因此这三列在 train 恒为 0、在 valid/test 才有非零值；LightGBM 会把它们纳入 schema，却无法从训练中学到序列通道价值，存在明确 train-serving feature shift。
- P1：SASRecF 序列已使用 user-day-item 去重购物篮语义，但正式规则候选、RRF 活跃度分层和已购过滤仍通过 `.inter` 最近 N 个交易行构造 history；同日多 SKU 和数量重复会改变历史长度、种子商品以及 activity tier。Industrial 的购物篮协议尚未贯穿全召回/排序链。
- P1：`build_cross_feature_table` 的 cohort history 只包含当前候选批次中出现的用户，而不是该 age bucket 的全部历史用户；所谓 cohort popularity 会依赖评估用户集合，并且快照用户本身是由未来标签周活跃性筛出的，口径不等同线上人群热度。
- P1（可运行性）：全量 raw 约 4.1GB，transactions CSV 约 3.2GB；ranking table 物化一次性读取 events/user features/item/customer features，拼接每用户最多 500 候选，并逐候选 Python 循环计算交叉特征。微型测试闭环成立，但当前复杂度和内存模式对 3100 万行全量数据风险很高，仓库内没有真实 full-run 产物证明其完成过。
- P2：PIT user features 会为全部 weekly train snapshots × 全部有历史用户生成行，而 LambdaRank 默认只消费最近 4 个 train snapshots，存在显著冗余计算和落盘。
- P2：LambdaRank 只选择 numeric dtype；item/customer 的 token 类别特征被静默排除，numeric ID（例如 product_code）反而可能作为连续数值输入，特征表达不理想。
- P2：训练只丢弃候选数少于 2 的 group，不丢弃零正例 group，也没有常量列/分布漂移 gate；候选覆盖不足时会产生大量无排序梯度的训练组。
- P2：RRF 搜索只优化 sequence/popular/category/item2item，repurchase/style/content 使用固定辅助权重；这是可用首版，但限制了扩展召回的效果上限。
- P2：union 500 先按多通道覆盖数、再按最佳 rank 选 item，可能挤掉强单通道独占候选；应依靠 candidate diagnostics/A-B 验证，不应直接假定该规则最优。
- P2：`--build-backtest` 只写多窗口切分/标签，不自动执行多窗口训练、选模和汇总；当前正式结论仍来自单窗口。
- P2：snapshot/evaluation 只覆盖未来标签周有购买的用户，适合 H&M next-week purchaser 离线协议，不等同线上全量用户曝光/点击场景。
- P2：同一 run-id 可以在配置一致时重跑，但 runner 不按 manifest 跳过已完成步骤，也不做 stage 原子提交；它是产物命名空间隔离，不是真正断点续跑。
- 验证：45 项定向测试通过；完整 `make check` 再次通过 251 项测试及 Baseline/Industrial/兼容 CLI smoke checks。未运行 3.2GB raw 上的完整训练。

## 2026-08-21 代码重复与冗余审计

- Python 源码共约 21,894 行；Baseline 9,468 行，Industrial 10,766 行，shared 只有 549 行。
- Baseline/Industrial 有 55 个同路径文件在仅替换应用命名空间后相似度 ≥95%，合计覆盖 Baseline 约 9,123 行（约 96%）。这说明当前“算法应用隔离”主要通过复制实现达成，而不是两套已经独立演进的算法。
- 其中 12 组、1,560 行是字节级完全相同文件，包括 item_features、manifest、candidate union、coverage/report、checkpoint 工具、paths/time 和接口 facade。
- 有意保留的重复：两套 pipeline/stages、CLI wrapper、profile 配置、SASRecF 配置路径和各自应用入口。这些文件承担协议所有权和未来独立演进，不应只因相似就直接合并。
- 高价值运行冗余：Baseline data 默认构建 `customer_features/customers.parquet` 和完整 `item_features/items.parquet`，但 Baseline SASRecF/RRF 正式 DAG 不消费这两份 ranking parquet；只需要 `hm_seq.item`。这会额外扫描 customers/articles 并占用磁盘。
- 高价值代码冗余：Baseline data service 仍包含 events/baskets/labels/PIT user/cross feature 全套参数和实现，外层 Baseline command 又明确拒绝这些参数。六个 Industrial-only data 模块约 1,289 行在正式 Baseline 协议中没有必要。
- 高价值代码冗余：Baseline 只允许 popular/category/item2item，但仍复制 repurchase/style/content 三路实现、注册表分支、Top-K 参数和辅助 RRF 权重；单独三个算法文件约 389 行，正式 Baseline CLI 不会启用。
- Baseline `offline_eval.py` 仍保留 LambdaRank 对照、ranker scored CSV 和 replacement gate 逻辑，而 Baseline evaluation wrapper 明确禁止 `--ranker-scored-csv`。这部分属于可裁剪的协议外代码。
- Baseline `evaluation/baseline_command.py`、`data/profile.py` 没有被 Baseline 应用命令面引用；Industrial 的 `evaluation/baseline_command.py` 只服务旧统一 CLI。
- 旧顶层 data/domain/evaluation/pipeline/ranking/recall/training/candidates 共 27 个兼容文件、424 行。它们目前由统一 CLI、旧测试和文档引用，不能立即删除，但应视为迁移债务而不是第三套正式实现。
- 兼容层存在“物理文件被 alias 遮蔽”的冗余：`fashionrec.data.command` 实际加载 Industrial service，`fashionrec.recall.registry` 实际加载 Industrial channel_registry，`fashionrec.training.command/checkpoint_command` 实际加载 Baseline service；对应 facade 文件本身不会执行，容易误导维护者。
- Industrial 的 `data/events.py`、`baskets.py`、`features.py` 只是 public re-export facade；目标目录测试要求其存在。更干净的做法是把 `build_events.py/build_baskets.py` 实现重命名到目标模块，而不是同时保留实现文件和别名文件。
- 三份 SASRecF YAML（baseline、industrial、legacy root）参数完全相同，仅首行注释不同。Baseline/Industrial 两份是有意独立所有权；root `configs/sasrecf.yaml` 仅为旧统一 CLI 兼容。
- `configs/sasrec.yaml` 是不同的 SASRec/BPR 手工实验配置，并非重复 SASRecF 配置；README 仍引用它，不能按重复文件删除。
- 应用内部另有小型重复：content/style 各自复制同一 `_load_interactions`；多个 CLI wrapper 只有 4～10 行。这些影响小，不应先于运行冗余处理。
- 推荐整理边界：若坚持“两套算法完全物理隔离”，不要把核心算法重新共享；先精简 Baseline 到真正四路/RRF 所需模块。若接受“应用隔离、稳定算法共享”，再把 exact-identical 的 paths/time/checkpoint/union/report 等下沉 shared，但这会改变此前的隔离原则。
- 验证：源码哈希、规范化 AST/diff、import reachability 和运行时模块身份检查完成；19 项隔离/公开入口测试通过，compileall 与 git diff check 通过。本轮未修改算法代码。

## 2026-08-21 冗余清理结果

- Baseline 数据准备现在只生成 `hm.inter`、时间切分、`hm.model_train.inter`、购物日因果序列、`hm_seq.item`、可选 backtest splits 与 manifest；不再扫描/写出 customer/item ranking parquet，也不再包含 events/baskets parquet/labels/PIT user/cross feature 分支。
- Baseline 删除了 Industrial-only 数据实现：events、basket parquet、labels、snapshots、user/cross/customer ranking features 和 raw profile；购物日序列所需的完整日截断函数已归入 `build_sequences.py`。
- Baseline 商品表改成紧凑实现，只读取 SASRecF 所需 8 个类别列，并仅保留序列 split 中出现的 SKU；不再先构造完整 24 字段 ranking feature table。
- Baseline 召回注册表和 CLI service 现在只支持 Popular、Category Popular、Item2Item；Repurchase、Style、Content 文件和辅助 RRF 权重已删除。
- Baseline offline evaluation 只保留行级购买集合 + Weighted RRF；next-basket label loader、LambdaRank CSV、replacement gate、comparison report 以及未接入 DAG 的候选诊断副本已删除。权重搜索也不再接受 labels_dir。
- Industrial 的 `events.py` 和 `baskets.py` 已直接承载真实实现，`build_events.py` / `build_baskets.py` 镜像文件删除；旧 `fashionrec.data.build_*` import 通过 alias 继续指向新 canonical 模块。
- 删除 10 个不会执行或没有引用的兼容 shadow / 旧 pipeline 子目录文件；统一 CLI 依赖的 package-level alias 和 profile registry 保留。
- 清理后 Baseline 从约 9,468 行降至 5,488 行，减少 3,980 行（约 42.0%）。规范化相似度 ≥95% 的同路径文件从 55 组降至 31 组，覆盖行数从约 9,123 降至 3,656；字节级完全相同从 12 组/1,560 行降至 8 组/697 行。
- 仍保留的重复主要是两应用独立拥有的 SASRecF、基础召回、RRF 与稳定基础工具；是否进一步下沉 shared 属于新的架构选择，不应在“算法物理隔离”约束下自动执行。
- 验证：完整 `make check` 通过，254 项测试全部收集并执行成功，Baseline/Industrial/旧统一 CLI smoke checks 通过；`git diff --check`、双 profile Make dry-run 和 import 边界检查通过。未运行 3.2GB raw 的真实全链训练。

## 2026-08-21 清理后双链路复审（进行中）

- Baseline 默认 DAG 仍是 10 步：data → train → checkpoint selection → valid recall/candidates → test recall/candidates → valid weight search → valid/test evaluation。所有子命令均指向 `fashionrec.baseline`。
- Industrial 默认 DAG 仍是 14 步：data → train → checkpoint selection → valid/test recall+candidates → ranking dataset → next-basket RRF weight search → LambdaRank train → valid/test predict → valid/test comparison evaluation。所有子命令均指向 `fashionrec.industrial`。
- Baseline 配置已收缩到本链实际消费字段；Industrial 保留购物篮/PIT/扩展召回/LambdaRank 全协议字段。两者都是 26+1+1 周、每通道 200、union 500、最终 Top-12，但标签和排序协议不同。
- 编排层的文件依赖表面闭环：Baseline candidates 消费本 run SASRecF recall，weights/evaluate 消费本 run candidates；Industrial ranker dataset 消费本 run data/candidates，train/predict/evaluate 依次消费本 run ranking artifacts。
- Industrial 训练候选仍只由 `ALL_CHANNELS` 的六路规则召回生成；valid/test 候选直接读取正式候选 CSV，并包含额外 SASRecF 通道。训练表固定声明 `ALL_CHANNELS + sasrecf`，因此 train 的 `sasrecf_present/score/rank` 为默认/常量，valid/test 才出现真实值，存在明确的 LambdaRank train-serving 特征分布偏移。
- LambdaRank 当前仅自动选择 numeric dtype 特征；token 类类别字段不会进入模型，numeric 编码的 ID/类别则会被当作连续数值。训练只删除少于 2 个候选的 group，没有删除零正例 group，也没有常量列或跨 split 漂移校验。
- Industrial cross feature 的 cohort 历史构造只遍历当前 `pairs` 批次里的用户，再从全局历史中过滤这些用户；因此 `item_cohort_purchase_count_*` 不是整个 age bucket 的历史热度，而是“当前候选批次中同桶用户”的热度，会随训练/评估用户集合变化。
- Industrial valid/test 规则召回和 RRF 活跃度仍通过 `.inter` 的最近 N 行构造 `user_history`。该函数不按 user-day-item 去重，也不按 basket 计数；是否污染各通道取决于底层召回器是否再次做日级聚合，需逐通道确认。
- SASRecF 数据准备本身是正确的购物日因果语义：先按 user-day-item 去重；同日多个目标共享完全相同的历史；训练仅在整日结束后推进；valid 周内不推进，valid 完成后整体加入 test 历史。因此“同日伪序列”已从两套 SASRecF 路径中消除。
- Industrial next-basket 标签按未来 7 天的 user-item 集合聚合，数量只保存在 `label_quantity`，MAP/Recall/NDCG 的 actual 使用 set，不会把同 SKU 数量重复计为多次命中；valid 权重搜索与 valid/test 最终评估都显式读取同一 run 的 labels，且搜权入口强制只允许 valid。
- 规则召回仍未完全采用购物篮协议：用户种子历史是最近 N 个 `.inter` 行；repurchase 的 count 也是交易行数；popular/category/style/content 的流行度统计也按交易行计数。Item2Item 默认 cosine/IUF 会对用户全窗口 SKU 去重，但其 `sequential` 变体仍可能把同日商品的确定性排序当转移顺序。数量作为购买强度可有业务意义，但当前与新标签的“去重篮子”语义不一致，且重复行会挤占最近种子/活跃度长度。
- 候选 union 500 的截断顺序是通道覆盖数优先、再最佳通道 rank、再最高原始 score；这能偏好多路共识，但可能挤掉强单通道独占候选，需要以候选覆盖/独占命中诊断验证，不能视为天然最优。
- Industrial 的 RRF 搜权只优化 SASRecF/Popular/CategoryPopular/Item2Item 四路，Repurchase/Style/Content 由活跃度模板追加固定辅助权重；流程可运行，但新增通道没有通过 valid 联合寻优。
- LambdaRank replacement gate 只要 MAP/Recall/NDCG 任一项提升且没有分层 MAP 相对下降超过 5%，就会给出 `replace_default_ranker=true`；它没有要求主指标 MAP@12 本身提升，也没有约束其他总体指标恶化。不过该 gate 当前只写建议，正式输出仍保持 RRF，不会自动切换。
- 全量运行仍未验证：raw 约 4.1GB、transactions CSV 约 3.2GB；ranking materialization 一次性读取 events/snapshots/labels/user/customer/item parquet，构建全量 candidate pairs 与 cross features，并对每个候选执行 Python 循环；train/valid parquet 又被 LightGBM 一次性读入。微型闭环通过不能证明 500 候选/用户的全量链可在现有机器内存和时限内完成。
- 两链预处理不会在切分前按未来总购买数筛用户或裁剪历史；可选 Top-30k 商品过滤也只在 train 窗拟合后冻结并应用到 valid/test，未发现该处标签泄漏。
- 本轮验证：70 项定向回归通过；完整 `make check` 通过，254 项测试全部收集执行成功且所有 Baseline/Industrial/兼容 CLI smoke checks 通过；双应用 dry-run、跨应用 import 扫描、`git diff --check` 通过。没有运行 3.2GB raw 的真实全链训练。
- 2026-08-21 修复结果：Industrial 新增 `data/basket_history.py` 作为唯一交易行→购物篮历史适配；`ranking/fusion.py`、`ranking/dataset_materialization.py` 均改用它，复购按购买日次数统计。
- 2026-08-21 修复结果：cohort 交叉特征使用完整 customer cohort；LambdaRank 训练/valid/test 统一六路规则候选，SASRecF 仍保留在 RRF 候选链但不再作为只有 valid/test 才有值的 ranker 特征；LambdaRank 排除零正例 group，并持久化 token 特征编码表。
- 2026-08-21 最终收口：Popular、CategoryPopular、Style、Content 的热度统计均先按 user-day-item 去重；Repurchase 的频次改为购买日次数。同日购买数量不会再通过重复交易行改变 Industrial 的热度或复购分数。
- 2026-08-21 最终收口：LambdaRank 若 valid 过滤零正例 group 后为空，会关闭该次 early-stopping eval；训练仍可用有效 train group 完成。token 映射被冻结到 schema，未知/缺失类别统一编码为 0。
- 2026-08-21 最终验证：261 项测试、两应用与兼容 CLI smoke、跨应用 import、编译与 diff whitespace 检查均通过。该结论仍是代码/微型数据闭环，不代表 3.2GB 原始交易数据上的全量训练耗时、内存和指标已经实测。
- 2026-08-21 SASRecF→LambdaRank 设计：现有正式 SASRecF checkpoint 使用完整 train 并只导出 valid/test；不能为更早 train snapshot 打分。正确接入必须为每个被 LambdaRank 消费的 train snapshot 构建截止 as-of 的序列 benchmark、独立训练 checkpoint，并给该快照用户导出 SASRecF 候选/分数。valid/test 继续消费正式 checkpoint，从而三类表都有同名 `sasrecf_present/score/rank` 且时间因果成立。
- 2026-08-21 SASRecF→LambdaRank 实现：Industrial 配置默认 `use_sequence_features=true`、最近 4 个快照、每快照最多 15 epochs、内部 7 天 validation。每个快照模型的数据词表、训练、验证和推理历史都不包含 as-of 之后事件。
- 2026-08-21 SASRecF→LambdaRank 实现：训练快照候选并集现在包含六路规则召回 + 当期 SASRecF；valid/test 保留正式候选中的 SASRecF。LightGBM 因而能学习序列通道的 present、原始 score、rank 以及其对 channel_count/best_rank/max_score 的贡献。
- 2026-08-21 资源影响：默认会在正式 SASRecF 之外额外训练 4 个快照 SASRecF，计算成本显著上升；代码测试通过不等于 3.2GB 全量数据已完成这些模型训练或已证明指标提升。

## 2026-08-21 Industrial 单 SASRecF 决策

- 用户明确不要快照级 SASRecF，选择“简单复用”：完整 train 训练并经正式 valid 选出的唯一 checkpoint，同时用于正式 valid/test 召回和历史 LambdaRank train 快照特征。
- 历史 train 快照推理仍按各自 as-of 截断用户输入序列，避免把未来购买直接塞进模型输入；但模型参数、商品词表和 checkpoint 选择看过该快照之后的数据，因此该协议不是严格 PIT，离线 LambdaRank 指标可能虚高。
- 实现应只加载一次 checkpoint、循环多个 train snapshot 推理；不再创建 `sasrecf_ranker_snapshots` checkpoint 目录，也不保留 `sequence_snapshot_epochs/sequence_validation_days` 等快照训练配置。
- 产物报告必须记录 `causal_model=false`、`history_as_of=true` 和唯一 `model_file`，避免后续把该实验误认为无泄漏工业评估。
- 最终实现只保留 `checkpoints/sasrecf/` shortlist 与 `checkpoints/sasrecf_selected.pth`；不存在 `sasrecf_ranker_snapshots`。valid/test 和历史 ranker train 证据都引用同一个 selected checkpoint。
- 历史 train 快照只重建内存中的 item sequence 并执行 full-sort 推理，不再为每个快照创建 RecBole dataset 或训练模型。训练成本从正式模型 + 4 个快照模型降为正式模型 1 次，仍需承担 4 个快照的推理成本。
- 当前代码验证证明单模型契约和表结构闭环，不证明实际指标提升；由于用户选择简单复用，任何 LambdaRank 离线提升都必须同时展示非严格 PIT 警告。

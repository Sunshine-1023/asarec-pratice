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

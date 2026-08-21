# FashionRec 双训练链路隔离与新协议修复

## Goal
修复当前审计发现的 P0/P1 链路问题，并把原始 RRF baseline 与购物篮/PIT/LambdaRank industrial 流程固化成两个独立 profile。两条链共享只读 raw 数据和代码，但配置、run 目录、处理中间数据、模型、候选、权重与评估产物互不覆盖。

## Phases
- [completed] 1. 固化修复边界与 ranking dataset 命令设计
- [completed] 2. 实现 next-basket/PIT 数据准备与 ranking table 物化
- [completed] 3. 修复召回 Top-K、复购和 RRF 多通道逻辑
- [completed] 4. 实现 baseline/industrial profile、产物命名空间和完整 pipeline 契约
- [completed] 5. 补充隔离/因果数据集测试、运行全量回归并更新文档
- [completed] 6. 提取 pipeline contracts 与无 profile 倾向的公共 stage builders
- [completed] 7. 将 baseline / industrial 编排拆入独立 package，并通过 registry 路由
- [completed] 8. 迁移 industrial ranking dataset、profile 配置和链路测试目录
- [completed] 9. 完整回归、CLI/Make smoke check 与结构文档收口
- [completed] 10. 冻结“两套应用 + 极小共享内核”目标与现有命令依赖图
- [completed] 11. 建立 shared kernel，并把稳定领域契约/纯指标迁入共享层
- [completed] 12. 建立 top-level baseline 应用，独立拥有 data/model/recall/ranking/evaluation/pipeline/CLI
- [completed] 13. 建立 top-level industrial 应用，独立拥有购物篮/PIT/扩展召回/LambdaRank/pipeline/CLI
- [completed] 14. 切换 Makefile/兼容入口、拆分两套模型配置并删除 profile 条件型正式入口
- [completed] 15. 加强应用边界测试、全量回归与文档收口
- [completed] 16. 冻结目标物理目录、兼容策略与模块迁移清单
- [completed] 17. 补齐 shared/interfaces 与 shared/io 中立契约
- [completed] 18. 将 baseline 数据、召回、RRF、评估与 SASRecF 实现归入 baseline
- [completed] 19. 将 industrial 购物篮/PIT、扩展召回、LambdaRank 与评估实现归入 industrial
- [completed] 20. 将旧顶层算法目录收缩为兼容 facade，并加强依赖边界测试
- [completed] 21. 运行全量回归、双链路 dry-run、文档与结构收口
- [completed] 22. 审计两套 DAG、命令参数和产物依赖闭环
- [completed] 23. 审计 Baseline 数据、训练、召回、RRF 与评估逻辑
- [completed] 24. 审计 Industrial 购物篮/PIT、因果候选、LambdaRank 与评估逻辑
- [completed] 25. 运行静态检查、定向回归并识别未覆盖的真实运行风险
- [completed] 26. 汇总完整性结论、问题严重级别和建议修复顺序
- [completed] 27. 扫描物理文件、兼容 facade 与双链路重复实现
- [completed] 28. 检查函数/配置/测试/入口的重复与死代码
- [completed] 29. 区分有意隔离、必要兼容和可删除冗余
- [completed] 30. 输出冗余清单、风险和整理顺序
- [completed] 31. 收缩 Baseline 数据协议并停止生成未消费的 ranking 特征
- [completed] 32. 收缩 Baseline 召回与评估到四路召回 + Weighted RRF
- [completed] 33. 整理 Industrial data 公共模块命名，移除 wrapper/build 双文件
- [completed] 34. 清理安全可退役的死文件与过期兼容引用
- [completed] 35. 运行定向回归、完整 make check、重复度复测与文档收口
- [completed] 36. 复核清理后 Baseline/Industrial 实际 DAG 与命令参数
- [completed] 37. 核对两链数据、标签、历史和防泄漏语义
- [completed] 38. 核对召回、候选、排序、权重和评估闭环
- [completed] 39. 运行结构/逻辑定向回归并复查全量测试状态
- [completed] 40. 输出两条具体流程、完整性判断和问题优先级
- [completed] 41. 统一 Industrial 购物篮历史协议
- [completed] 42. 修正 cohort 全桶 PIT 聚合
- [completed] 43. 消除 LambdaRank train-serving 特征偏移
- [completed] 44. 定向回归、全量检查与文档收口
- [completed] 45. 统一 Industrial 热度/复购的同日数量去重并补齐回归锁定
- [completed] 46. 修复 LambdaRank 空验证集边界并完成最终结构/全量验证
- [completed] 47. 设计并物化训练快照级因果 SASRecF 召回产物
- [completed] 48. 将 SASRecF present/score/rank 接入 LambdaRank train/valid/test 表
- [completed] 49. 接入 Industrial 独立 DAG、配置、产物契约与 CLI
- [completed] 50. 补充防泄漏/特征一致性测试并完成全量回归
- [completed] 51. 将 Industrial SASRecF 协议改为单一正式 checkpoint 简单复用
- [completed] 52. 用单模型为多个 LambdaRank train 快照生成序列召回证据
- [completed] 53. 删除快照级训练配置、额外 checkpoint 与误导性文档
- [completed] 54. 补充泄漏标识/单模型契约测试并完成全量回归

## Decisions
- `ranking.enabled: false` 时保持当前默认 RRF 链的执行成本和产物语义，不强制扫描额外事件/特征。
- `ranking.enabled: true` 时由 pipeline 显式开启 labels、PIT user/cross features，并在 ranker 训练前生成固定 ranking parquet。
- LambdaRank 训练样本必须只使用 as-of 及以前历史；禁止用训练完成后的 SASRecF 为更早快照生成会泄漏的序列分数。
- 用户于 2026-08-21 明确选择“简单复用”：Industrial 只训练一个正式 SASRecF，历史 LambdaRank train 快照复用该 checkpoint；保留 as-of 用户输入历史，但接受模型参数、商品词表和 checkpoint 选择看过后续数据造成的非严格因果离线指标。
- RRF 继续作为安全基线，但要么显式支持所有已物化通道，要么不得让零权重通道无声参与 union 截断。
- 首版使用最近若干个 train weekly snapshots；数量由 `ranking.train_snapshot_limit` 显式配置，默认 4，避免全量重复构建 25 套规则索引。
- 正式产物路径改为 `outputs/runs/<profile>/<run_id>/`；两个 profile 即使使用相同 run-id 也不发生覆盖。
- baseline profile 强制 `ranking.enabled=false` 且只物化原四路（3 路规则 + SASRecF）；industrial profile 强制 `ranking.enabled=true` 并物化扩展召回与 LambdaRank。
- `--skip-ranker` 只跳过 ranking dataset/train/predict 执行，不改变 industrial 的数据和评估协议；这样分阶段命令仍使用 next-basket 标签。
- 本轮只重构代码所有权和目录，不修改两条 DAG 的步骤顺序、参数、模型或指标语义。
- data / recall / ranking / evaluation / training 保持共享组件层；禁止复制成 baseline/industrial 两份。
- profile 差异只允许存在于 `pipeline/baseline` 和 `pipeline/industrial`，公共 stage builder 不读取 `ranking.enabled`。
- 新目标依赖方向为 `baseline|industrial application -> shared kernel/algorithm libraries`；两应用之间禁止互相导入。
- 两应用使用独立模块入口 `python -m fashionrec.baseline` 与 `python -m fashionrec.industrial`，Makefile 不再通过通用 pipeline 的 `--profile` 选择正式链路。
- 允许共享无业务语义的领域契约、纯指标、通用算法实现和执行器；数据协议、命令入口、召回集合、模型配置、排序/评估协议与 DAG 必须归属各自应用。
- 本轮按用户给出的物理目录落位；旧 `fashionrec.data/recall/ranking/training/evaluation` 只允许保留向新位置转发的兼容 facade，不再承载正式实现。
- 为避免一次重构改变实验结果，baseline 与 industrial 可以各自拥有同内容的首版算法实现；后续优化分别演进，跨链共享仅限 `shared` 中无业务倾向的契约、I/O、指标和运行时。
- 清理策略采用“删 Baseline 协议外能力，不回收两应用核心算法到 shared”；保留两套 SASRecF/三路基础召回/RRF 的独立所有权。
- Baseline `hm_seq.item` 改为只读取 SASRecF 所需 8 个类别字段和序列中实际出现的 SKU，不再构建完整 ranking item table。
- 旧 import 兼容继续通过 package-level `sys.modules` alias 提供；已被 alias 遮蔽的同名物理 facade 文件直接删除，避免第三份假实现。

## Next Step
单 SASRecF 简单复用链路已完成；下一步用新 run-id 真实运行 Industrial，测量 8GB CUDA 下的耗时、显存和 LambdaRank 指标。

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| 旧 planning 文件仍描述 2026-08-18 只读审查 | 1 | 已按本轮修复任务重建计划并保留审计结论到 findings.md |
| ranker 编排测试仍断言第一步直接训练 | 1 | 新增 dataset 物化后属于过期断言，将随双链路 DAG 测试一起更新 |
| 单个补丁同时删除并新增 `pipeline/orchestrator.py`，被 `apply_patch` 拒绝 | 1 | 补丁未产生文件改动；改为按 contracts、common、profile、facade 分批迁移 |
| 新 baseline DAG 与旧计划首次等价性断言失败 | 1 | 原因是新旧 dataclass 类型不同；按 `(name, command)` 比较后 baseline/industrial 多组计划完全一致 |
| 测试移动到子目录后仍按 `parents[1]` 解析仓库根目录 | 1 | integration/pipeline 测试统一改为 `parents[2]`，避免 Make 在 tests/ 下执行和空目录扫描假通过 |
| shared kernel 首次兼容层补丁匹配了错误的 `domain/__init__.py` 文本 | 1 | 文件移动和 import 替换已完成；改为按实际文件内容分小补丁新增兼容 facade |
| 切换兼容 facade 的补丁再次同时 delete/add 同一路径 | 1 | 补丁原子失败无残留；改为先更新 registry/删除旧文件，再单独新增 facade |
| 应用入口切换后 15 项旧结构测试失败 | 1 | 均为旧通用 CLI/`--profile`/shim import 断言；恢复步骤名称，测试改为验证 top-level 应用及固定协议 wrapper |
| shared contracts 移动后被文件系统命名为 `contracts 2.py` | 1 | 收集阶段立即发现；重命名为标准 `contracts.py` 后重新运行全量回归 |
| 直接运行 `python -m fashionrec.baseline|industrial --help` 找不到包 | 1 | 当前仓库未 editable install，后续 smoke check 使用项目既有的 `PYTHONPATH=src`/Make 环境；compileall 已通过 |
| 兼容包首次只替换 `__path__`，同一 dataclass 被按两个模块名加载 | 1 | 改成 `sys.modules` 别名，使旧路径与应用路径引用同一模块对象，恢复类型相等和 monkeypatch 语义 |
| 大批定向测试运行到 LightGBM predict 时原生层 segmentation fault | 1 | 定位为 Recall 兼容包过早加载 Torch；恢复 SASRecF 延迟导入后组合测试与完整 `make check` 均通过 |
| 本轮尝试用 ruff/pyflakes 做静态冗余检查，但环境未安装 | 1 | 改用 AST import/reachability、源码哈希、规范化 diff 和运行时模块身份检查完成审计 |
| 定向回归命令误写不存在的 `tests/test_fusion.py`、`tests/test_sequence_preparation.py`、`tests/test_build_item_features.py` | 1 | 用 `rg --files tests` 定位真实文件名后重新运行，相关测试全部通过 |
| Industrial alias 迁移补丁首次按错误的字典顺序匹配 `data/__init__.py` | 1 | 原子补丁未产生改动；读取实际内容后按真实顺序重新应用成功 |
| 同日数量去重测试首次断言并列商品的 rank-normalized 分数相等 | 1 | 并列计数仍按 item ID 稳定赋 rank；改为比较重复输入与去重输入的完整索引一致性 |
| `compileall` 在 Industrial 源码树生成 `__pycache__` | 1 | 仅清除本次检查生成的字节码缓存，未触碰源码和实验产物；后续验证使用 `PYTHONDONTWRITEBYTECODE=1` |
| 用 `--dry-run` 检查 Industrial pipeline，但该 CLI 不提供该参数 | 1 | 命令在参数解析阶段退出、未执行步骤；后续改用 `build_pipeline_steps` 做只读 DAG 检查 |
| 单次补丁同时匹配 progress/findings 尾部失败 | 1 | 补丁原子失败、无文件改动；读取实际尾部后分文件追加。 |

"""Data preparation: preprocess, split, causal sequences, and item features."""  # 数据准备（预处理、划分、序列化、商品特征）

from __future__ import annotations  # 启用延迟注解评估

import argparse  # 导入命令行参数解析模块
import sys  # 导入系统模块以处理中断退出
import time  # 导入时间模块以统计各步骤耗时
from pathlib import Path  # 导入路径处理类

from fashionrec.industrial.data.baskets import BASKET_SCHEMA_VERSION, build_baskets  # 可选按天购物篮
from fashionrec.industrial.data.events import EVENT_SCHEMA_VERSION, build_events  # 可选 user-day-item 事件
from fashionrec.industrial.data.build_item_features import build_item_features  # 导入商品特征构建函数
from fashionrec.industrial.data.user_features import USER_FEATURE_SCHEMA_VERSION, build_user_features  # as-of 行为特征
from fashionrec.industrial.data.cross_features import CROSS_FEATURE_SCHEMA_VERSION, build_cross_features  # 用户×商品交叉
from fashionrec.industrial.data.customer_features import (  # 用户静态画像
    CUSTOMER_FEATURE_SCHEMA_VERSION,
    DEFAULT_CUSTOMERS,
    build_customer_features,
)
from fashionrec.industrial.data.item_features import ITEM_FEATURE_SCHEMA_VERSION  # 全量商品特征语义
from fashionrec.industrial.data.build_sequences import prepare_recbole_benchmark_files, read_max_item_list_length  # 数据层序列构建
from fashionrec.industrial.data.labels import LABEL_SCHEMA_VERSION, build_labels  # 可选 next-basket 标签
from fashionrec.industrial.data.snapshots import SNAPSHOT_SCHEMA_VERSION  # 快照索引语义
from fashionrec.industrial.data.filter import run_filter  # 导入原始数据过滤函数
from fashionrec.industrial.data.manifest import SCHEMA_VERSION, build_processed_hm_manifest, write_manifest  # 数据快照清单
from fashionrec.industrial.data.preprocess import (  # 数据预处理与显式输入路径
    MAX_USER_HISTORY,
    MIN_USER_PURCHASES,
    RAW_PATH,
    WEEKS,
    build_inter_file,
)
from fashionrec.industrial.data.backtest import (  # 可选多窗口回测
    BACKTEST_SCHEMA_VERSION,  # 回测语义
    DEFAULT_N_WINDOWS,  # 默认 3 窗
    build_backtest_windows,  # 写出各窗口切分
    required_preprocess_weeks,  # 拉长 hm.inter
    window_split_paths,  # 单窗口目录
)
from fashionrec.industrial.data.split import (  # 时间划分与 train-only 模型子集
    TEST_WEEKS,
    TRAIN_WEEKS,
    VALID_WEEKS,
    build_model_train_split,
    split_bounds_dict,
    split_by_time,
)
from fashionrec.shared.experiment.config import load_experiment_config  # 可选统一实验协议

DEFAULT_CONFIG = Path("configs/industrial/models/sasrecf.yaml")
DEFAULT_PROCESSED_DIR = Path("data/processed")  # 独立调用时的旧目录；流水线应传入 run-scoped 路径


def processed_layout(processed_dir: Path) -> dict[str, Path]:  # 一次运行内处理后数据布局
    root = Path(processed_dir)  # 规范化
    hm = root / "hm"  # RecBole 交互
    seq = root / "hm_seq"  # 序列文件
    return {  # 逻辑名到路径
        "root": root,  # 根
        "hm": hm,  # hm
        "seq": seq,  # 序列
        "filtered": root / "filtered",  # 本 run 新鲜 filtered，禁止回落到 data/raw/filtered
        "inter": hm / "hm.inter",  # 全量交互
        "train": hm / "hm.train.inter",  # 训练
        "model_train": hm / "hm.model_train.inter",  # 模型拟合子集
        "valid": hm / "hm.valid.inter",  # 验证
        "test": hm / "hm.test.inter",  # 测试
        "seq_item": seq / "hm_seq.item",  # 商品特征
        "events": root / "events",  # 同日同 SKU 事件，按月分区 parquet
        "baskets": root / "baskets",  # 按天购物篮，按月分区 parquet
        "snapshots": root / "snapshots",  # (user, as_of_date) 样本索引
        "labels": root / "labels",  # next-basket 去重标签
        "backtest": root / "backtest",  # 多窗口切分；默认不写
        "item_features": root / "item_features" / "items.parquet",  # SKU/款式静态特征
        "customer_features": root / "customer_features" / "customers.parquet",  # 用户静态画像
        "user_features": root / "user_features",  # as-of 行为特征，按 as_of_date 分区
        "cross_features": root / "cross_features",  # 用户×商品交叉，按 as_of_date 分区
        "manifest": root / "manifest.json",  # 数据清单
    }  # 布局结束


def select_transactions_input(*, with_filter: bool, filtered_path: Path | None = None) -> Path:  # 显式决定本次交易输入
    if not with_filter:  # 默认永远走 raw
        return RAW_PATH  # 禁止因磁盘上已有 filtered 而自动切换
    if filtered_path is None:  # 没有本 run 刚写出的 filtered
        raise ValueError(  # 拒绝复用全局旧产物
            "with_filter=True requires filtered_path produced in this run; "
            "refusing to reuse data/raw/filtered/"
        )  # 错误信息
    path = Path(filtered_path)  # 规范化
    if not path.is_file():  # 本 run 未真正写出
        raise FileNotFoundError(f"filtered_path does not exist: {path}")  # 失败而不是回落
    return path  # 只接受本次生成的路径


def _run_step(name: str, fn) -> None:  # 执行单个数据准备子步骤并打印耗时
    print(f"\n{'=' * 60}")  # 打印步骤分隔线
    print(f"[{name}]")  # 打印步骤名称
    print("=" * 60)  # 打印分隔线结束
    started = time.perf_counter()  # 记录步骤开始时间
    fn()  # 调用子步骤函数
    elapsed = time.perf_counter() - started  # 计算步骤耗时
    print(f"Done in {elapsed:.1f}s")  # 打印步骤完成与耗时


def main(argv: list[str] | None = None) -> None:  # 命令行入口：按顺序执行数据准备流程
    parser = argparse.ArgumentParser(  # 创建参数解析器
        prog="fashionrec data",  # 统一 CLI 子命令名
        description="Run data preparation steps in order for SASRecF / offline eval.",  # 程序描述
    )  # 参数解析器创建结束
    parser.add_argument(  # 定义 --with-filter 参数
        "--with-filter",  # 参数名
        action="store_true",  # 布尔开关
        help="Create and use a fresh train-fitted filtered dataset in --processed-dir/filtered; never reuse data/raw/filtered/.",
    )  # --with-filter 参数结束
    parser.add_argument(  # 定义 --config 参数
        "--config",  # 参数名
        type=Path,  # 路径类型
        default=DEFAULT_CONFIG,  # 默认配置文件
        help=f"Config for MAX_ITEM_LIST_LENGTH when building hm_seq (default: {DEFAULT_CONFIG})",  # 帮助文本
    )  # --config 参数结束
    parser.add_argument(  # 定义 --skip-item-features 参数
        "--skip-item-features",  # 参数名
        action="store_true",  # 布尔开关
        help="Skip hm_seq conversion and hm_seq.item (offline eval only needs hm.*).",  # 帮助文本
    )  # --skip-item-features 参数结束
    parser.add_argument(  # 统一实验协议，只影响数据窗口/阈值并写入 manifest
        "--experiment-config",  # 参数名
        type=Path,  # 路径类型
        default=None,  # 默认不强制
        help="Optional unified experiment YAML; records protocol into <processed-dir>/manifest.json",  # 帮助文本
    )  # --experiment-config 结束
    parser.add_argument(  # 本次处理后数据目录；流水线传入 outputs/runs/<profile>/<run_id>/data
        "--processed-dir",  # 参数名
        type=Path,  # 路径类型
        default=None,  # 未指定时回退旧目录，便于独立调试
        help="Write processed artifacts here instead of overwriting data/processed. Pipeline must pass the run data directory.",
    )  # --processed-dir 结束
    parser.add_argument(  # 默认跳过；阶段 1.1 事件表不替换 hm.inter
        "--build-events",  # 参数名
        action="store_true",  # 布尔开关
        help="Write monthly-partitioned user-day-item events to --processed-dir/events; off by default.",
    )  # --build-events 结束
    parser.add_argument(  # 默认跳过全量购物篮落盘；序列构建本身已按日共享历史
        "--build-baskets",  # 参数名
        action="store_true",  # 布尔开关
        help="Write monthly-partitioned daily baskets to --processed-dir/baskets; off by default.",
    )  # --build-baskets 结束
    parser.add_argument(  # 默认跳过；切分之后才写，避免训练标签吃到 valid/test
        "--build-labels",  # 参数名
        action="store_true",  # 布尔开关
        help="Write weekly snapshot index and next-basket labels to --processed-dir/{snapshots,labels}; off by default.",
    )  # --build-labels 结束
    parser.add_argument(  # 默认跳过；打开后写出 3 个时间窗口，不训练三遍模型
        "--build-backtest",  # 参数名
        action="store_true",  # 布尔开关
        help="Write rolling backtest splits under --processed-dir/backtest; official test is window 0 only.",
    )  # --build-backtest 结束
    parser.add_argument(  # 默认跳过；需要事件/交易 + 切分边界
        "--build-user-features",
        action="store_true",
        help="Write point-in-time user behavior features under --processed-dir/user_features; off by default.",
    )  # --build-user-features 结束
    parser.add_argument(  # 默认跳过；需要 labels 或 --candidates
        "--build-cross-features",
        action="store_true",
        help="Write user-item cross features under --processed-dir/cross_features; requires --build-labels or --candidates.",
    )  # --build-cross-features 结束
    parser.add_argument(  # 显式候选对（无标签时）
        "--candidates",
        type=Path,
        default=None,
        help="CSV with user_id,item_id,as_of_date for cross features when labels are not built.",
    )  # --candidates 结束
    parser.add_argument(  # 用户主数据
        "--customers",
        type=Path,
        default=DEFAULT_CUSTOMERS,
        help="customers.csv for static user features.",
    )  # --customers 结束
    parser.add_argument(  # 同款新色标签需要 articles
        "--articles",  # 参数名
        type=Path,  # 路径
        default=Path("data/raw/articles.csv"),  # 默认主数据
        help="articles.csv for product_code when --build-labels is set.",
    )  # --articles 结束
    args = parser.parse_args(argv)  # 解析显式命令参数

    config_path = args.config  # 获取配置文件路径
    if not config_path.exists():  # 配置文件不存在
        raise FileNotFoundError(f"Config not found: {config_path}")  # 抛出文件未找到错误

    experiment = None  # 可选实验协议
    if args.experiment_config is not None:  # 若提供了统一配置
        experiment = load_experiment_config(args.experiment_config)  # 加载并校验

    processed_dir = Path(args.processed_dir) if args.processed_dir is not None else DEFAULT_PROCESSED_DIR  # 产物根
    layout = processed_layout(processed_dir)  # 固定本次写出路径
    layout["root"].mkdir(parents=True, exist_ok=True)  # 确保目录存在

    protocol_weeks = experiment.data.total_weeks if experiment is not None else WEEKS  # 单窗口协议周数
    history_weeks = experiment.data.history_weeks if experiment is not None else TRAIN_WEEKS  # 训练历史
    valid_weeks = experiment.data.valid_weeks if experiment is not None else VALID_WEEKS  # 验证周
    test_weeks = experiment.data.test_weeks if experiment is not None else TEST_WEEKS  # 测试周
    n_windows = experiment.data.backtest_windows if experiment is not None else DEFAULT_N_WINDOWS  # 回测窗数
    weeks = protocol_weeks  # 默认预处理窗口等于协议
    if args.build_backtest:  # 早期窗口需要更长历史，否则会被截断
        weeks = required_preprocess_weeks(  # 协议周 + (窗数-1) 周
            train_weeks=history_weeks,  # 训练
            valid_weeks=valid_weeks,  # 验证
            test_weeks=test_weeks,  # 测试
            n_windows=n_windows,  # 窗口数
        )  # 拉长结束
    min_user_purchases = experiment.data.min_user_purchases if experiment is not None else MIN_USER_PURCHASES  # 最少购买
    max_user_history = experiment.data.max_user_history if experiment is not None else MAX_USER_HISTORY  # 历史上限
    keep_full_item_universe = (  # 默认保留全量目录 SKU，Top-30k 只走 --with-filter
        experiment.data.keep_full_item_universe if experiment is not None else True
    )
    customers_input = Path(args.customers)  # 默认 raw customers
    split_result = None  # 保存切分边界供 manifest 使用
    transactions_input = RAW_PATH  # 默认 raw；with-filter 时覆盖为本 run 产物

    if args.with_filter:  # 若指定先运行过滤
        filtered_dir = layout["filtered"]  # 本 run 的 filtered，不写 data/raw/filtered
        _run_step(  # 执行过滤步骤
            "1/6 causal item sampling",  # 步骤名
            lambda: run_filter(  # 按协议过滤
                min_user_purchases=min_user_purchases,  # 最少购买
                max_user_behaviors=max_user_history,  # 每用户最大行为
                weeks=weeks,  # 周数
                valid_weeks=experiment.data.valid_weeks if experiment is not None else 1,
                test_weeks=experiment.data.test_weeks if experiment is not None else 1,
                output_dir=filtered_dir,  # 只写本次目录
            ),  # 过滤调用结束
        )  # 过滤步骤结束
        transactions_input = select_transactions_input(  # 只接受刚写出的文件
            with_filter=True,  # 过滤模式
            filtered_path=filtered_dir / "transactions_train.csv",  # 本 run 产物
        )  # 选择结束
        filtered_customers = filtered_dir / "customers.csv"  # 本 run 过滤用户
        if filtered_customers.is_file():  # 与交易同源
            customers_input = filtered_customers  # 用 filtered customers
        step = 2  # 下一步从 2 开始编号
    else:  # 未指定过滤
        transactions_input = select_transactions_input(with_filter=False)  # 永远 raw
        step = 1  # 从步骤 1 开始编号

    if args.build_events:  # 显式请求才写事件表，默认跳过以免误跑 3100 万行
        _run_step(  # 与 hm.inter 并行的新语义产物
            "build user-day-item events",  # 步骤名
            lambda: build_events(  # 聚合并按月分区
                transactions_path=transactions_input,  # 与 preprocess 同一输入
                output_dir=layout["events"],  # 写到本次 events/
            ),  # 构建结束
        )  # 事件步骤结束

    if args.build_baskets:  # 显式请求才写购物篮表
        _run_step(  # 按天成篮
            "build daily baskets",  # 步骤名
            lambda: build_baskets(  # 有事件则复用，否则从交易现算
                output_dir=layout["baskets"],  # 本次 baskets/
                events_dir=layout["events"] if args.build_events else None,  # 本 run 事件
                transactions_path=None if args.build_events else transactions_input,  # 无事件时用同一输入
            ),  # 构建结束
        )  # 购物篮步骤结束

    _run_step(  # 执行预处理
        f"{step}/6 preprocess",  # 步骤名
        lambda: build_inter_file(  # 构建交互文件
            transactions_path=transactions_input,  # 显式使用 raw 或本次 filtered 产物
            output_path=layout["inter"],  # 写到本次 processed-dir
            weeks=weeks,  # 周数
            min_user_purchases=min_user_purchases,  # 最少购买
            max_user_history=max_user_history,  # 历史上限
        ),  # 预处理结束
    )  # 预处理步骤结束
    step += 1  # 步骤编号加一

    def _split() -> None:  # 闭包：按协议切分并记住边界
        nonlocal split_result  # 写外层变量
        split_result = split_by_time(  # 官方窗口仍按协议周数，不用拉长后的预处理周数
            inter_path=layout["inter"],  # 本 run 的 hm.inter
            train_inter_path=layout["train"],  # 训练
            valid_inter_path=layout["valid"],  # 验证
            test_inter_path=layout["test"],  # 测试
            total_weeks=protocol_weeks,  # 协议总周，不是 backtest 拉长周数
            train_weeks=history_weeks,  # 训练周
            valid_weeks=valid_weeks,  # 验证周
            test_weeks=test_weeks,  # 测试周
        )  # 切分结束

    _run_step(f"{step}/6 split", _split)  # 执行按时间划分
    step += 1  # 步骤编号加一

    _run_step(  # 只用 train 统计模型训练用户，不删除完整历史
        f"{step}/6 model_train",
        lambda: build_model_train_split(
            train_path=layout["train"],  # 本 run 训练交互
            output_path=layout["model_train"],  # 本 run 模型训练子集
            min_user_purchases=min_user_purchases,  # 最少购买
        ),
    )
    step += 1

    inter_paths_for_users = (layout["train"], layout["valid"], layout["test"])  # 切分用户并集
    _run_step(  # 2.2 用户静态特征，不依赖 hm_seq
        f"{step}/6 build_customer_features",
        lambda: build_customer_features(
            customers_path=customers_input,  # raw 或本 run filtered
            output_path=layout["customer_features"],  # parquet
            inter_paths=inter_paths_for_users,  # 补齐 unknown
            keep_full_customer_universe=True,  # 保留全量 customers 目录
        ),
    )
    step += 1

    if args.build_labels:  # 切分完成后才写标签，训练快照不会吃到 valid/test 周
        if split_result is None:  # 没有时间边界就不能生成 weekly 快照
            raise RuntimeError("split must finish before --build-labels")  # 失败
        horizon_days = experiment.label.horizon_days if experiment is not None else 7  # 默认 7 天
        target_mode = experiment.label.target_mode if experiment is not None else "next_basket"  # 默认购物篮
        _run_step(  # next-basket 标签
            "build next-basket labels",  # 步骤名
            lambda: build_labels(  # 样本索引 + 去重标签
                split=split_result,  # 时间切分
                snapshots_dir=layout["snapshots"],  # 索引目录
                labels_dir=layout["labels"],  # 标签目录
                horizon_days=horizon_days,  # 未来窗口
                events_dir=layout["events"] if args.build_events else None,  # 复用本 run 事件
                transactions_path=None if args.build_events else transactions_input,  # 否则从交易现算
                articles_path=args.articles,  # 款式映射
                target_mode=target_mode,  # 目前只支持 next_basket
            ),  # 构建结束
        )  # 标签步骤结束

    if args.build_user_features:  # 2.3 as-of 行为特征
        if split_result is None:  # 无切分边界
            raise RuntimeError("split must finish before --build-user-features")  # 失败
        horizon_days = experiment.label.horizon_days if experiment is not None else 7  # 默认 7 天
        _run_step(  # 每个 snapshot 只用 as_of 及以前历史
            "build user behavior features",
            lambda: build_user_features(
                split=split_result,  # 时间切分
                output_dir=layout["user_features"],  # 分区 parquet
                horizon_days=horizon_days,  # 快照日历
                events_dir=layout["events"] if args.build_events else None,  # 复用事件
                transactions_path=None if args.build_events else transactions_input,  # 否则交易
                articles_path=args.articles,  # 品类/颜色/款式
            ),
        )  # 用户行为特征结束

    if args.build_cross_features:  # 2.4 用户×商品交叉
        labels_dir = layout["labels"] if layout["labels"].exists() or args.build_labels else None  # 标签对
        if labels_dir is None and args.candidates is None:  # 无对来源
            raise RuntimeError("--build-cross-features requires --build-labels (or existing labels/) or --candidates")  # 失败
        _run_step(
            "build user-item cross features",
            lambda: build_cross_features(
                output_dir=layout["cross_features"],
                events_dir=layout["events"] if args.build_events else None,
                transactions_path=None if args.build_events else transactions_input,
                articles_path=args.articles,
                labels_dir=labels_dir,
                candidates_path=args.candidates,
                customers_path=customers_input,
            ),
        )  # 交叉特征结束

    if args.build_backtest:  # 显式请求才写多窗口，默认不训三遍、不写三份切分
        if split_result is None:  # 没有官方锚点
            raise RuntimeError("split must finish before --build-backtest")  # 失败
        horizon_days = experiment.label.horizon_days if experiment is not None else 7  # 默认 7 天
        target_mode = experiment.label.target_mode if experiment is not None else "next_basket"  # 默认购物篮

        def _build_backtest() -> None:  # 闭包：切分各窗口，可选再写标签
            written = build_backtest_windows(  # 从同一份 hm.inter 切出 w0/w1/w2
                inter_path=layout["inter"],  # 已拉长的交互
                output_dir=layout["backtest"],  # 本次 backtest/
                train_weeks=history_weeks,  # 训练
                valid_weeks=valid_weeks,  # 验证
                test_weeks=test_weeks,  # 测试
                n_windows=n_windows,  # 窗口数
                max_date=split_result.max_date,  # 与官方切分同一锚点
            )  # 切分结束
            if not args.build_labels and not args.build_user_features and not args.build_cross_features:  # 只切分
                return  # 结束
            for window, window_split in written:  # 每窗口标签/特征
                paths = window_split_paths(layout["backtest"], window.window_id)  # 窗口目录
                window_labels_dir = paths["labels"]  # 窗口标签目录
                if args.build_labels:  # next-basket 标签
                    build_labels(  # 训练标签不会吃到该窗口的 valid/test
                        split=window_split,  # 该窗口时间切分
                        snapshots_dir=paths["snapshots"],  # 窗口索引
                        labels_dir=window_labels_dir,  # 窗口标签
                        horizon_days=horizon_days,  # 未来窗口
                        events_dir=layout["events"] if args.build_events else None,  # 复用本 run 事件
                        transactions_path=None if args.build_events else transactions_input,  # 否则从交易现算
                        articles_path=args.articles,  # 款式映射
                        target_mode=target_mode,  # 目前只支持 next_basket
                    )  # 标签结束
                if args.build_user_features:  # 每窗口 as-of 行为
                    build_user_features(
                        split=window_split,  # 窗口切分
                        output_dir=paths["root"] / "user_features",  # 窗口内目录
                        horizon_days=horizon_days,  # 快照
                        events_dir=layout["events"] if args.build_events else None,  # 事件
                        transactions_path=None if args.build_events else transactions_input,  # 交易
                        articles_path=args.articles,  # 商品
                    )
                if args.build_cross_features:  # 每窗口交叉特征
                    build_cross_features(
                        output_dir=paths["root"] / "cross_features",
                        events_dir=layout["events"] if args.build_events else None,
                        transactions_path=None if args.build_events else transactions_input,
                        articles_path=args.articles,
                        labels_dir=window_labels_dir if (args.build_labels or window_labels_dir.exists()) else None,
                        candidates_path=args.candidates,
                        customers_path=customers_input,
                    )

        _run_step("build backtest windows", _build_backtest)  # 多窗口切分

    def _write_manifest() -> None:  # 数据准备成功后写快照
        preprocess = {  # 记录实际使用的预处理参数
            "schema_version": SCHEMA_VERSION,  # 当前行级交互语义
            "weeks": weeks,  # 实际写入 hm.inter 的周数（开回测时会拉长）
            "protocol_weeks": protocol_weeks,  # 单窗口 train+valid+test
            "min_user_purchases": min_user_purchases,  # 最少购买
            "max_user_history": max_user_history,  # 历史上限
            "with_filter": bool(args.with_filter),  # 是否先过滤
            "transactions_input": str(transactions_input),  # 本次明确选择的输入路径
            "customers_input": str(customers_input),  # 用户主数据路径
            "processed_dir": str(processed_dir),  # 本次产物目录
            "skip_item_features": bool(args.skip_item_features),  # 是否跳过序列特征
            "sasrec_config": str(config_path),  # 序列配置
            "experiment_config": str(args.experiment_config) if args.experiment_config else None,  # 实验协议
            "experiment_name": experiment.experiment.name if experiment is not None else None,  # 实验名
            "seed": experiment.experiment.seed if experiment is not None else None,  # 种子
            "keep_full_item_universe": keep_full_item_universe,  # 全量 SKU
            "deduplicate_user_day_item": experiment.data.deduplicate_user_day_item if experiment is not None else None,  # 同日去重
            "label_target_mode": experiment.label.target_mode if experiment is not None else None,  # 标签语义
            "label_horizon_days": experiment.label.horizon_days if experiment is not None else None,  # 标签窗口
            "ranking_enabled": experiment.ranking.enabled if experiment is not None else None,  # 学习排序开关
            "build_events": bool(args.build_events),  # 是否写出事件表
            "events_schema_version": EVENT_SCHEMA_VERSION if args.build_events else None,  # 事件语义
            "events_dir": str(layout["events"]) if args.build_events else None,  # 事件目录
            "build_baskets": bool(args.build_baskets),  # 是否写出购物篮
            "baskets_schema_version": BASKET_SCHEMA_VERSION if args.build_baskets else None,  # 购物篮语义
            "baskets_dir": str(layout["baskets"]) if args.build_baskets else None,  # 购物篮目录
            "build_labels": bool(args.build_labels),  # 是否写出 next-basket 标签
            "snapshots_schema_version": SNAPSHOT_SCHEMA_VERSION if args.build_labels else None,  # 快照语义
            "labels_schema_version": LABEL_SCHEMA_VERSION if args.build_labels else None,  # 标签语义
            "snapshots_dir": str(layout["snapshots"]) if args.build_labels else None,  # 快照目录
            "labels_dir": str(layout["labels"]) if args.build_labels else None,  # 标签目录
            "build_backtest": bool(args.build_backtest),  # 是否写出多窗口回测
            "backtest_schema_version": BACKTEST_SCHEMA_VERSION if args.build_backtest else None,  # 回测语义
            "backtest_dir": str(layout["backtest"]) if args.build_backtest else None,  # 回测目录
            "backtest_windows": n_windows if args.build_backtest else None,  # 窗口数
            "item_features_schema_version": None if args.skip_item_features else ITEM_FEATURE_SCHEMA_VERSION,  # 商品特征
            "item_features_path": None if args.skip_item_features else str(layout["item_features"]),  # parquet
            "customer_features_schema_version": CUSTOMER_FEATURE_SCHEMA_VERSION,  # 用户特征
            "customer_features_path": str(layout["customer_features"]),  # parquet
            "build_user_features": bool(args.build_user_features),  # 是否写出 as-of 行为
            "user_features_schema_version": USER_FEATURE_SCHEMA_VERSION if args.build_user_features else None,  # 语义
            "user_features_dir": str(layout["user_features"]) if args.build_user_features else None,  # 目录
            "build_cross_features": bool(args.build_cross_features),  # 是否写出交叉特征
            "cross_features_schema_version": CROSS_FEATURE_SCHEMA_VERSION if args.build_cross_features else None,  # 语义
            "cross_features_dir": str(layout["cross_features"]) if args.build_cross_features else None,  # 目录
            "cross_features_candidates_path": str(args.candidates) if args.candidates is not None else None,  # 显式候选
        }  # 预处理记录结束
        payload = build_processed_hm_manifest(  # 流式统计处理后文件
            processed_dir=processed_dir,  # 不扫描全局 data/processed
            raw_transactions=transactions_input,  # 与 build_inter_file 完全相同的实际输入
            true_raw_transactions=RAW_PATH,  # 始终记录真正 raw
            preprocess=preprocess,  # 参数
            split_bounds=split_bounds_dict(split_result) if split_result is not None else {},  # 时间边界
            repo_root=Path.cwd(),  # 从项目根读取 Git SHA
        )  # 清单结束
        out = write_manifest(payload, layout["manifest"])  # 写到本次 processed-dir
        print(f"Wrote data manifest: {out}")  # 提示路径

    if args.skip_item_features:  # 若跳过序列化与商品特征
        print("\nSkipped hm_seq + build_item_features (--skip-item-features).")  # 提示已跳过
        _write_manifest()  # 仍写出数据清单
        return  # 提前结束

    max_item_list_length = read_max_item_list_length(config_path)  # 从配置读取最大序列长度

    def _prepare_seq() -> None:  # 闭包：准备 hm_seq 序列文件
        prepare_recbole_benchmark_files(  # 调用 RecBole 基准文件准备
            max_item_list_length,  # 展平序列长度
            train_split_file=layout["model_train"],  # 模型训练
            valid_split_file=layout["valid"],  # 验证
            test_split_file=layout["test"],  # 测试
            target_dir=layout["seq"],  # 写到本次 hm_seq
            train_history_file=layout["train"],  # 完整训练历史
            max_shopping_days=max_user_history if experiment is not None else None,  # 最近 N 个购物日
        )  # 序列准备结束

    _run_step(f"{step}/6 hm_seq", _prepare_seq)  # 执行 hm_seq 转换
    step += 1  # 步骤编号加一
    _run_step(  # 执行商品特征构建
        f"{step}/6 build_item_features",
        lambda: build_item_features(
            articles_path=args.articles,  # 与标签同一份 articles
            output_path=layout["seq_item"],  # 本 run 的 RecBole item 文件
            inter_paths=(  # 本 run 的序列划分
                layout["seq"] / "hm_seq.train.inter",
                layout["seq"] / "hm_seq.valid.inter",
                layout["seq"] / "hm_seq.test.inter",
            ),
            features_output_path=layout["item_features"],  # 全量 SKU/款式 parquet
            keep_full_item_universe=keep_full_item_universe,  # 默认不截断目录
        ),
    )  # 商品特征结束

    print("\nData preparation finished.")  # 提示数据准备完成
    _write_manifest()  # 写出数据快照清单
    print("Next: make train RUN_ID=<id>")  # 推荐通过统一 Makefile 入口启动训练


if __name__ == "__main__":  # 脚本直接运行时
    try:  # 捕获键盘中断
        main()  # 调用主函数
    except KeyboardInterrupt:  # 用户 Ctrl+C 中断
        print("\nInterrupted.", file=sys.stderr)  # 向 stderr 打印中断提示
        sys.exit(130)  # 以标准中断退出码退出

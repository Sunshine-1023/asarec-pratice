"""Build RecBole item feature file for hm_seq dataset."""  # 构建 hm_seq 数据集的 RecBole 商品特征文件

from __future__ import annotations  # 启用延迟注解

import argparse  # 命令行参数解析
from pathlib import Path  # 路径对象

import pandas as pd  # 读交互商品 ID

from fashionrec.data.item_features import (  # 全量商品特征
    RECB_CATEGORY_COLUMNS,  # SASRecF 仍用的 8 个类别
    RECB_ITEM_FILE_COLUMNS,  # RecBole 列
    build_item_feature_table,  # SKU + 款式表
    clean_category_token,  # 类别清洗，供召回复用
    load_articles_table,  # 读 articles
    recbole_item_frame,  # 切 RecBole 切片
    write_item_features_parquet,  # 排序用 parquet
)
from fashionrec.domain.ids import canonical_item_id  # 统一商品 ID 契约


RAW_ARTICLES_PATH = Path("data/raw/articles.csv")  # H&M 商品元数据原始文件
SEQ_DATASET_DIR = Path("data/processed/hm_seq")  # hm_seq 数据目录
OUTPUT_ITEM_FILE = SEQ_DATASET_DIR / "hm_seq.item"  # RecBole 商品特征输出文件
SEQ_SPLIT_FILES = (  # 序列训练/验证/测试文件
    SEQ_DATASET_DIR / "hm_seq.train.inter",  # 训练集交互文件
    SEQ_DATASET_DIR / "hm_seq.valid.inter",  # 验证集交互文件
    SEQ_DATASET_DIR / "hm_seq.test.inter",  # 测试集交互文件
)  # 结束序列划分文件元组

RAW_FEATURE_COLUMNS = list(RECB_CATEGORY_COLUMNS)  # 兼容旧导入：8 个类别名
ITEM_FILE_COLUMNS = list(RECB_ITEM_FILE_COLUMNS)  # 兼容旧导入：RecBole 列


def _collect_inter_items(inter_paths: list[Path]) -> set[str]:  # 收集 hm_seq.*.inter 中出现的商品 ID
    item_ids: set[str] = set()  # 商品 ID 集合
    for inter_path in inter_paths:  # 遍历每个划分文件
        if not inter_path.exists():  # 文件不存在
            raise FileNotFoundError(f"Missing split file: {inter_path}")  # 提示先构建序列切分文件
        df = pd.read_csv(inter_path, sep="\t", usecols=["item_id:token"], dtype={"item_id:token": "string"})  # 读取目标列
        normalized = df["item_id:token"].map(canonical_item_id)  # 标准化商品 ID
        item_ids.update(item_id for item_id in normalized if item_id)  # 加入集合
    return item_ids  # 返回集合


def build_item_features(  # 构建 hm_seq.item，并可选写出全量 parquet
    articles_path: Path = RAW_ARTICLES_PATH,  # 商品元数据输入路径
    output_path: Path = OUTPUT_ITEM_FILE,  # RecBole item 文件路径
    inter_paths: tuple[Path, ...] = SEQ_SPLIT_FILES,  # 交互划分文件路径集合
    features_output_path: Path | None = None,  # 全量 SKU parquet；默认不写
    keep_full_item_universe: bool = True,  # 默认不按 Top-30k / 仅交互商品截断目录
) -> Path:  # 返回 RecBole 输出路径
    inter_item_ids = _collect_inter_items(list(inter_paths))  # 收集交互中商品 ID
    if not inter_item_ids:  # 交互中没有商品
        raise ValueError("No item ids found in hm_seq split files.")  # 报错提示

    articles_df = load_articles_table(articles_path)  # 读全量 24 字段，缺列当缺失
    features = build_item_feature_table(  # 目录 SKU + 交互缺失 unknown
        articles_df,  # 主数据
        extra_item_ids=inter_item_ids,  # 序列里出现过的
        keep_full_item_universe=keep_full_item_universe,  # 是否保留从未交易的目录 SKU
    )  # 特征表结束
    recb = recbole_item_frame(features, item_ids=inter_item_ids)  # RecBole 只覆盖序列商品
    missing_metadata = int((features["is_unknown_item:float"] == 1.0).sum())  # unknown 行数

    output_path = Path(output_path)  # 规范化
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    recb.to_csv(output_path, sep="\t", index=False)  # 写出 RecBole item 文件
    print(f"saved: {output_path}")  # RecBole 路径
    print(f"rows: {len(recb):,}")  # RecBole 行数
    print(f"covered item ids from inter: {len(inter_item_ids):,}")  # 交互商品数
    print(f"missing metadata backfilled: {missing_metadata:,}")  # 补齐 unknown

    if features_output_path is not None:  # 写出排序用全量表
        written = write_item_features_parquet(features, features_output_path)  # parquet
        print(f"saved item features: {written} ({len(features):,} rows)")  # 提示

    return output_path  # 返回 RecBole 路径


def main() -> None:  # CLI 入口
    parser = argparse.ArgumentParser(description="Build hm_seq.item for RecBole SASRecF training")  # 参数解析器
    parser.add_argument("--articles-path", type=Path, default=RAW_ARTICLES_PATH)  # 商品元数据路径
    parser.add_argument("--output-path", type=Path, default=OUTPUT_ITEM_FILE)  # item 文件输出路径
    parser.add_argument("--train-inter-path", type=Path, default=SEQ_SPLIT_FILES[0])  # 训练划分路径
    parser.add_argument("--valid-inter-path", type=Path, default=SEQ_SPLIT_FILES[1])  # 验证划分路径
    parser.add_argument("--test-inter-path", type=Path, default=SEQ_SPLIT_FILES[2])  # 测试划分路径
    parser.add_argument(  # 全量特征 parquet
        "--features-output-path",
        type=Path,
        default=None,
        help="Write full SKU/style parquet here; omitted means RecBole file only.",
    )
    parser.add_argument(  # 快速实验才关掉全量目录
        "--no-full-item-universe",
        action="store_true",
        help="Restrict parquet to items seen in hm_seq splits (still backfills unknown metadata).",
    )
    args = parser.parse_args()  # 解析命令行参数

    build_item_features(  # 生成 hm_seq.item
        articles_path=args.articles_path,  # 输入商品元数据
        output_path=args.output_path,  # 输出 item 路径
        inter_paths=(args.train_inter_path, args.valid_inter_path, args.test_inter_path),  # 三个 split 文件
        features_output_path=args.features_output_path,  # 可选 parquet
        keep_full_item_universe=not args.no_full_item_universe,  # 默认全量
    )  # 结束 build_item_features 调用


if __name__ == "__main__":  # 直接执行脚本
    main()  # 调用入口

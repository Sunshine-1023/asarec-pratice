"""Build RecBole item feature file for hm_seq dataset."""  # 构建 hm_seq 数据集的 RecBole 商品特征文件

from __future__ import annotations  # 启用延迟注解

import argparse  # 命令行参数解析
import re  # 正则表达式处理
from pathlib import Path  # 路径对象

import pandas as pd  # 数据处理


RAW_ARTICLES_PATH = Path("data/raw/articles.csv")  # H&M 商品元数据原始文件
SEQ_DATASET_DIR = Path("data/processed/hm_seq")  # hm_seq 数据目录
OUTPUT_ITEM_FILE = SEQ_DATASET_DIR / "hm_seq.item"  # RecBole 商品特征输出文件
SEQ_SPLIT_FILES = (  # 序列训练/验证/测试文件
    SEQ_DATASET_DIR / "hm_seq.train.inter",
    SEQ_DATASET_DIR / "hm_seq.valid.inter",
    SEQ_DATASET_DIR / "hm_seq.test.inter",
)

RAW_FEATURE_COLUMNS = [  # 从 articles.csv 读取的原始类别字段
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
]
ITEM_FILE_COLUMNS = [  # RecBole .item 文件输出字段（含类型后缀）
    "item_id:token",
    "product_type_name:token",
    "product_group_name:token",
    "colour_group_name:token",
    "section_name:token",
    "garment_group_name:token",
]


def _normalize_item_id(value: object) -> str:  # 统一商品 ID 为 10 位字符串
    text = str(value).strip()  # 转字符串并去掉首尾空白
    text = re.sub(r"\.0+$", "", text)  # 去除可能的浮点后缀
    return text.zfill(10)  # 左侧补零到 10 位


def _clean_category_token(value: object) -> str:  # 清洗类别字段为安全 token
    if pd.isna(value):  # 缺失值
        return "unknown"  # 缺失值统一填充
    text = str(value).strip()  # 转字符串并去掉首尾空白
    if not text:  # 空字符串
        return "unknown"  # 空值填充
    text = re.sub(r"[\s/\\]+", "_", text)  # 将空白、斜杠等分隔符替换为下划线
    text = re.sub(r"_+", "_", text).strip("_")  # 合并重复下划线并去首尾下划线
    return text or "unknown"  # 兜底为 unknown


def _collect_inter_items(inter_paths: list[Path]) -> set[str]:  # 收集 hm_seq.*.inter 中出现的商品 ID
    item_ids: set[str] = set()  # 商品 ID 集合
    for inter_path in inter_paths:  # 遍历每个划分文件
        if not inter_path.exists():  # 文件不存在
            raise FileNotFoundError(f"Missing split file: {inter_path}")  # 提示先构建序列切分文件
        df = pd.read_csv(inter_path, sep="\t", usecols=["item_id:token"], dtype={"item_id:token": "string"})  # 读取目标列
        normalized = df["item_id:token"].map(_normalize_item_id)  # 标准化商品 ID
        item_ids.update(item_id for item_id in normalized if item_id)  # 加入集合
    return item_ids  # 返回集合


def build_item_features(  # 构建 hm_seq.item 文件
    articles_path: Path = RAW_ARTICLES_PATH,  # 商品元数据输入路径
    output_path: Path = OUTPUT_ITEM_FILE,  # 输出 item 文件路径
    inter_paths: tuple[Path, ...] = SEQ_SPLIT_FILES,  # 交互划分文件路径集合
) -> Path:  # 返回输出路径
    if not articles_path.exists():  # articles.csv 不存在
        raise FileNotFoundError(f"articles.csv not found: {articles_path}")  # 报错

    inter_item_ids = _collect_inter_items(list(inter_paths))  # 收集交互中商品 ID
    if not inter_item_ids:  # 交互中没有商品
        raise ValueError("No item ids found in hm_seq split files.")  # 报错提示

    articles_df = pd.read_csv(articles_path, dtype={"article_id": "string"})  # 读取商品元数据
    required_columns = {"article_id", *RAW_FEATURE_COLUMNS}  # 必需列集合
    missing_columns = required_columns.difference(articles_df.columns)  # 缺失列集合
    if missing_columns:  # 若有缺失列
        missing = ", ".join(sorted(missing_columns))  # 拼接缺失列名
        raise ValueError(f"articles.csv missing required columns: {missing}")  # 报错

    feature_df = articles_df[["article_id", *RAW_FEATURE_COLUMNS]].copy()  # 仅保留所需列
    feature_df["item_id:token"] = feature_df["article_id"].map(_normalize_item_id)  # 生成标准化 item_id
    feature_df = feature_df.drop(columns=["article_id"])  # 删除原始 article_id
    feature_df = feature_df.rename(columns={col: f"{col}:token" for col in RAW_FEATURE_COLUMNS})  # 重命名特征列
    feature_df = feature_df[feature_df["item_id:token"].isin(inter_item_ids)]  # 仅保留交互中出现的商品
    feature_df = feature_df.drop_duplicates(subset=["item_id:token"], keep="first")  # 商品去重

    for col in ITEM_FILE_COLUMNS[1:]:  # 清洗所有类别字段
        feature_df[col] = feature_df[col].map(_clean_category_token)  # 执行字段清洗

    existing_item_ids = set(feature_df["item_id:token"])  # 当前已覆盖商品集合
    missing_item_ids = sorted(inter_item_ids - existing_item_ids)  # 元数据中缺失的商品
    if missing_item_ids:  # 对缺失商品补 unknown 行，保证与 inter 对齐
        missing_df = pd.DataFrame({"item_id:token": missing_item_ids})  # 构造缺失商品表
        for col in ITEM_FILE_COLUMNS[1:]:  # 全部类别特征赋值 unknown
            missing_df[col] = "unknown"  # 缺失特征填充
        feature_df = pd.concat([feature_df, missing_df], ignore_index=True)  # 合并回完整结果

    feature_df = feature_df[ITEM_FILE_COLUMNS].sort_values("item_id:token").reset_index(drop=True)  # 调整列顺序并排序
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    feature_df.to_csv(output_path, sep="\t", index=False)  # 写出 RecBole item 文件

    print(f"saved: {output_path}")  # 输出保存路径
    print(f"rows: {len(feature_df):,}")  # 输出行数
    print(f"covered item ids from inter: {len(inter_item_ids):,}")  # 输出覆盖的交互商品数
    print(f"missing metadata backfilled: {len(missing_item_ids):,}")  # 输出补齐 unknown 数量
    return output_path  # 返回输出路径


def main() -> None:  # CLI 入口
    parser = argparse.ArgumentParser(description="Build hm_seq.item for RecBole SASRecF training")  # 参数解析器
    parser.add_argument("--articles-path", type=Path, default=RAW_ARTICLES_PATH)  # 商品元数据路径
    parser.add_argument("--output-path", type=Path, default=OUTPUT_ITEM_FILE)  # item 文件输出路径
    parser.add_argument("--train-inter-path", type=Path, default=SEQ_SPLIT_FILES[0])  # 训练划分路径
    parser.add_argument("--valid-inter-path", type=Path, default=SEQ_SPLIT_FILES[1])  # 验证划分路径
    parser.add_argument("--test-inter-path", type=Path, default=SEQ_SPLIT_FILES[2])  # 测试划分路径
    args = parser.parse_args()  # 解析命令行参数

    build_item_features(  # 生成 hm_seq.item
        articles_path=args.articles_path,  # 输入商品元数据
        output_path=args.output_path,  # 输出 item 路径
        inter_paths=(args.train_inter_path, args.valid_inter_path, args.test_inter_path),  # 三个 split 文件
    )


if __name__ == "__main__":  # 直接执行脚本
    main()  # 调用入口

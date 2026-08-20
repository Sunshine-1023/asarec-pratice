"""SKU and product_code item features for ranking; RecBole still uses a category slice."""  # 商品静态特征：全量字段 + 款式变体

from __future__ import annotations  # 延迟注解

import re  # 文本与类别清洗
from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.domain.ids import canonical_item_id  # 十位 SKU


ITEM_FEATURE_SCHEMA_VERSION = "hm.item_features.v1"  # 特征语义
UNKNOWN_TOKEN = "unknown"  # 缺失类别/未见商品
_FLOAT_SUFFIX = re.compile(r"\.0+$")  # CSV 数字列可能带 .0
_SPACE = re.compile(r"\s+")  # 连续空白

# H&M articles.csv 去掉 article_id 后的 24 个字段
ARTICLE_SOURCE_COLUMNS = (
    "product_code",
    "prod_name",
    "product_type_no",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_no",
    "graphical_appearance_name",
    "colour_group_code",
    "colour_group_name",
    "perceived_colour_value_id",
    "perceived_colour_value_name",
    "perceived_colour_master_id",
    "perceived_colour_master_name",
    "department_no",
    "department_name",
    "index_code",
    "index_name",
    "index_group_no",
    "index_group_name",
    "section_no",
    "section_name",
    "garment_group_no",
    "garment_group_name",
    "detail_desc",
)

# 需要同时保留 numeric / token 的 ID 列
NUMERIC_ID_COLUMNS = (
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_group_no",
    "section_no",
    "garment_group_no",
)

# 类别名称与非纯数字编码，只做 token
NAME_TOKEN_COLUMNS = (
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_code",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
)

TEXT_COLUMNS = ("prod_name", "detail_desc")  # 清洗后保留长度、词数、缺失标记

# SASRecF / 类别热门仍使用的 8 个类别 token
RECB_CATEGORY_COLUMNS = (
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "department_name",
    "index_name",
    "index_group_name",
)
RECB_ITEM_FILE_COLUMNS = ("item_id:token", *(f"{col}:token" for col in RECB_CATEGORY_COLUMNS))


def clean_category_token(value: object) -> str:  # 清洗类别字段为安全 token
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):  # 缺失
        return UNKNOWN_TOKEN  # 统一 unknown
    text = str(value).strip()  # 去空白
    if not text or text.lower() in {"nan", "<na>", "none"}:  # 空或占位
        return UNKNOWN_TOKEN  # unknown
    text = re.sub(r"[\s/\\]+", "_", text)  # 分隔符变下划线
    text = re.sub(r"_+", "_", text).strip("_")  # 合并下划线
    return text or UNKNOWN_TOKEN  # 兜底


def clean_text(value: object) -> str:  # 文本字段：小写、压空白，缺失变空串
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):  # 缺失
        return ""  # 空
    text = str(value).strip().lower()  # 小写
    if not text or text in {"nan", "<na>", "none"}:  # 占位
        return ""  # 空
    return _SPACE.sub(" ", text)  # 压空白


def _numeric_id_parts(value: object) -> tuple[float, str, float]:  # 数值、token、缺失标记
    token = clean_category_token(value)  # 先做 token
    if token == UNKNOWN_TOKEN:  # 没有有效 ID
        return 0.0, UNKNOWN_TOKEN, 1.0  # 数值 0，记缺失
    raw = str(value).strip()  # 原始文本
    numeric_text = _FLOAT_SUFFIX.sub("", raw)  # 去掉 .0
    try:  # 能转数字的 ID
        number = float(numeric_text)  # 解析
    except ValueError:  # 如 index 类编码
        return 0.0, token, 0.0  # 有值但非数字，不算缺失
    if pd.isna(number):  # 解析成 NaN
        return 0.0, UNKNOWN_TOKEN, 1.0  # 当缺失
    return float(number), token, 0.0  # 双形态


def _text_parts(value: object) -> tuple[str, float, float, float]:  # token、字数、词数、缺失
    cleaned = clean_text(value)  # 清洗
    if not cleaned:  # 空
        return UNKNOWN_TOKEN, 0.0, 0.0, 1.0  # 缺失
    n_tokens = float(len(cleaned.split(" ")))  # 空白分词
    return cleaned, float(len(cleaned)), n_tokens, 0.0  # 保留原文 token 供后续文本召回


def _combo_token(*values: object) -> str:  # 颜色/图案/材质组合
    parts = [clean_category_token(value) for value in values]  # 各段
    kept = [part for part in parts if part != UNKNOWN_TOKEN]  # 丢掉缺失段
    if not kept:  # 全缺
        return UNKNOWN_TOKEN  # unknown
    return "__".join(kept)  # 稳定拼接


def unknown_item_row(item_id: str) -> dict[str, object]:  # 未见 metadata 的 SKU，禁止静默丢弃
    row: dict[str, object] = {  # 一行
        "item_id": canonical_item_id(item_id),  # 十位
        "is_unknown_item:float": 1.0,  # 标记补齐
        "feature_version": ITEM_FEATURE_SCHEMA_VERSION,  # 版本
        "n_style_variants:float": 0.0,  # 无款式可数
        "colour_combo:token": UNKNOWN_TOKEN,  # 颜色组合
        "pattern_combo:token": UNKNOWN_TOKEN,  # 图案
        "shade_combo:token": UNKNOWN_TOKEN,  # 明暗
        "material_combo:token": UNKNOWN_TOKEN,  # 材质代理
    }  # 基础结束
    for col in NUMERIC_ID_COLUMNS:  # ID 双形态
        row[f"{col}:float"] = 0.0  # 数值
        row[f"{col}:token"] = UNKNOWN_TOKEN  # token
        row[f"{col}_missing:float"] = 1.0  # 缺失
    for col in NAME_TOKEN_COLUMNS:  # 名称
        row[f"{col}:token"] = UNKNOWN_TOKEN  # unknown
    for col in TEXT_COLUMNS:  # 文本
        row[f"{col}:token"] = UNKNOWN_TOKEN  # unknown
        row[f"{col}_n_chars:float"] = 0.0  # 长度
        row[f"{col}_n_tokens:float"] = 0.0  # 词数
        row[f"{col}_missing:float"] = 1.0  # 缺失
    return row  # 返回


def load_articles_table(articles_path: Path) -> pd.DataFrame:  # 读 articles，缺列当缺失而不是报错删行
    articles_path = Path(articles_path)  # 规范化
    if not articles_path.is_file():  # 缺文件
        raise FileNotFoundError(f"articles.csv not found: {articles_path}")  # 无法构建
    frame = pd.read_csv(articles_path, dtype="string")  # 全当字符串，保前导零
    if "article_id" not in frame.columns:  # 主键
        raise ValueError("articles must contain article_id")  # 报错
    for col in ARTICLE_SOURCE_COLUMNS:  # 24 字段
        if col not in frame.columns:  # 旧 fixture 可能缺列
            frame[col] = pd.NA  # 补空，后续变 unknown
    frame["item_id"] = frame["article_id"].map(canonical_item_id)  # 十位 SKU
    frame = frame.drop_duplicates("item_id", keep="first")  # 一 SKU 一行
    return frame  # 返回


def build_item_feature_table(  # 全量 SKU 特征；inter 里没有 metadata 的补 unknown
    articles: pd.DataFrame,  # 已 load 的 articles
    *,
    extra_item_ids: set[str] | None = None,  # 序列/召回里出现但主数据没有的 SKU
    keep_full_item_universe: bool = True,  # False 时只留 extra + 其 metadata，不丢 unknown
) -> pd.DataFrame:  # 特征表
    if "item_id" not in articles.columns:  # 需要规范化 ID
        raise ValueError("articles table must contain item_id")  # 报错
    extra = {canonical_item_id(item_id) for item_id in (extra_item_ids or set())}  # 规范化
    source = articles.copy()  # 不改调用方
    if not keep_full_item_universe:  # 快速实验：不必留下从未出现的目录 SKU
        if extra:  # 有交互商品
            source = source[source["item_id"].isin(extra)].copy()  # 只留出现过的
        else:  # 没有交互却要求截断
            source = source.iloc[0:0].copy()  # 空表，后面 extra 仍会补 unknown
    rows: list[dict[str, object]] = []  # 逐行组装，便于 unknown 模板复用
    for raw in source.to_dict(orient="records"):  # 目录内 SKU
        item_id = canonical_item_id(raw["item_id"])  # 十位
        row: dict[str, object] = {  # 一行特征
            "item_id": item_id,  # SKU
            "is_unknown_item:float": 0.0,  # 有主数据
            "feature_version": ITEM_FEATURE_SCHEMA_VERSION,  # 版本
            "colour_combo:token": _combo_token(  # 颜色组 + 主色
                raw.get("colour_group_name"),  # 颜色组
                raw.get("perceived_colour_master_name"),  # 主色
            ),
            "pattern_combo:token": _combo_token(raw.get("graphical_appearance_name")),  # 图案
            "shade_combo:token": _combo_token(raw.get("perceived_colour_value_name")),  # 明暗
            "material_combo:token": _combo_token(  # 没有独立材质列，用组别近似
                raw.get("product_group_name"),  # 产品组
                raw.get("garment_group_name"),  # 服装组
            ),
        }  # 组合结束
        for col in NUMERIC_ID_COLUMNS:  # ID 双形态
            number, token, missing = _numeric_id_parts(raw.get(col))  # 解析
            row[f"{col}:float"] = number  # 数值
            row[f"{col}:token"] = token  # token
            row[f"{col}_missing:float"] = missing  # 缺失
        for col in NAME_TOKEN_COLUMNS:  # 名称
            row[f"{col}:token"] = clean_category_token(raw.get(col))  # token
        for col in TEXT_COLUMNS:  # 文本
            token, n_chars, n_tokens, missing = _text_parts(raw.get(col))  # 统计
            row[f"{col}:token"] = token  # 清洗文本
            row[f"{col}_n_chars:float"] = n_chars  # 字符数
            row[f"{col}_n_tokens:float"] = n_tokens  # 词数
            row[f"{col}_missing:float"] = missing  # 缺失
        rows.append(row)  # 追加
    catalog = pd.DataFrame(rows)  # 目录表
    if catalog.empty:  # 没有主数据行
        catalog = pd.DataFrame(columns=list(unknown_item_row("0").keys()))  # 空表仍有列
        catalog["item_id"] = catalog["item_id"].astype("string")  # ID 类型
    else:  # 按款式数变体
        known = catalog["product_code:token"].ne(UNKNOWN_TOKEN)  # 有款式
        variant_counts = catalog.loc[known].groupby("product_code:token")["item_id"].transform("size")  # 同款 SKU 数
        catalog["n_style_variants:float"] = 1.0  # 无款式时至少自己
        catalog.loc[known, "n_style_variants:float"] = variant_counts.astype("float64")  # 有款式用计数
    present = set(catalog["item_id"].astype(str))  # 已有 SKU
    missing_ids = sorted(extra - present)  # 交互有、主数据无
    if missing_ids:  # 必须补 unknown，不能删
        unknown_df = pd.DataFrame([unknown_item_row(item_id) for item_id in missing_ids])  # 补齐
        catalog = pd.concat([catalog, unknown_df], ignore_index=True)  # 合并
    catalog = catalog.sort_values("item_id", kind="mergesort").reset_index(drop=True)  # 稳定排序
    return catalog  # 返回


def recbole_item_frame(features: pd.DataFrame, item_ids: set[str] | None = None) -> pd.DataFrame:  # SASRecF 切片
    frame = features.copy()  # 不改原表
    frame["item_id:token"] = frame["item_id"].map(canonical_item_id)  # RecBole 列名
    if item_ids is not None:  # 只覆盖序列里出现的商品
        wanted = {canonical_item_id(item_id) for item_id in item_ids}  # 规范化
        frame = frame[frame["item_id:token"].isin(wanted)].copy()  # 过滤
        missing = wanted - set(frame["item_id:token"])  # 仍缺的（不应发生）
        if missing:  # 再补 unknown
            extra = pd.DataFrame([unknown_item_row(item_id) for item_id in sorted(missing)])  # 补
            extra["item_id:token"] = extra["item_id"]  # RecBole ID
            frame = pd.concat([frame, extra], ignore_index=True)  # 合并
    recb = frame.loc[:, list(RECB_ITEM_FILE_COLUMNS)].copy()  # 只要 8 个类别
    recb = recb.drop_duplicates("item_id:token", keep="first")  # 去重
    return recb.sort_values("item_id:token", kind="mergesort").reset_index(drop=True)  # 排序


def write_item_features_parquet(features: pd.DataFrame, output_path: Path) -> Path:  # 写出排序用特征
    output_path = Path(output_path)  # 规范化
    if output_path.suffix == "":  # 传入的是目录
        output_path = output_path / "items.parquet"  # 默认文件名
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 建目录
    features.to_parquet(output_path, index=False, engine="pyarrow")  # parquet
    return output_path  # 返回

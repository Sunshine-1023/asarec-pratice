"""User static features from customers.csv for cold-start and ranking."""  # 用户静态画像

from __future__ import annotations  # 延迟注解

import hashlib  # postal_code 稳定哈希分桶
from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.industrial.data.item_features import UNKNOWN_TOKEN, clean_category_token  # 复用 token 清洗
from fashionrec.shared.domain.ids import canonical_user_id  # 用户 ID


CUSTOMER_FEATURE_SCHEMA_VERSION = "hm.customer_features.v1"  # 特征语义
DEFAULT_CUSTOMERS = Path("data/raw/customers.csv")  # 默认用户主数据
POSTAL_HASH_BUCKETS = 100  # 哈希分桶数，不做连续距离

CUSTOMER_SOURCE_COLUMNS = (  # customers.csv 除 customer_id 外的字段
    "FN",
    "Active",
    "club_member_status",
    "fashion_news_frequency",
    "age",
    "postal_code",
)

# 年龄分桶：闭区间右端 inclusive
AGE_BUCKET_BOUNDS: tuple[tuple[int, int, str], ...] = (
    (0, 17, "under_18"),
    (18, 24, "18_24"),
    (25, 34, "25_34"),
    (35, 44, "35_44"),
    (45, 54, "45_54"),
    (55, 64, "55_64"),
    (65, 120, "65_plus"),
)


def _is_missing(value: object) -> bool:  # 真缺失 vs 空字符串
    if value is None:  # None
        return True  # 缺失
    if isinstance(value, float) and pd.isna(value):  # NaN
        return True  # 缺失
    if pd.isna(value):  # pd.NA
        return True  # 缺失
    return False  # 有值（含空串）


def parse_binary_flag(value: object) -> tuple[float, float]:  # FN/Active：1、0/空、missing 仅 unknown 用户
    if _is_missing(value):  # CSV 空单元格在 H&M 里 = 明确未订阅
        return 0.0, 0.0  # 0/空缺态，不是 missing indicator
    text = str(value).strip()  # 原始
    if not text or text.lower() in {"nan", "<na>", "none"}:  # 空串
        return 0.0, 0.0  # 0/空缺态
    if text in {"1", "1.0", "true", "True", "YES", "yes", "Y", "y"}:  # 明确为 1
        return 1.0, 0.0  # 1 态
    if text in {"0", "0.0", "false", "False", "NO", "no", "N", "n"}:  # 明确为 0
        return 0.0, 0.0  # 0 态
    return 0.0, 0.0  # 其它非标准值当明确 0


def parse_age(value: object) -> tuple[float, str, float]:  # age、分桶、缺失
    if _is_missing(value):  # 缺失
        return 0.0, UNKNOWN_TOKEN, 1.0  # 不填均值
    try:  # 转数值
        age = float(str(value).strip())  # 解析
    except ValueError:  # 非法
        return 0.0, UNKNOWN_TOKEN, 1.0  # 当缺失
    if pd.isna(age) or age < 0 or age > 120:  # 越界
        return 0.0, UNKNOWN_TOKEN, 1.0  # 当缺失
    bucket = age_bucket_token(int(round(age)))  # 分桶
    return float(age), bucket, 0.0  # 返回


def age_bucket_token(age: int) -> str:  # 由年龄得到 bucket token
    for lo, hi, label in AGE_BUCKET_BOUNDS:  # 闭区间
        if lo <= age <= hi:  # 命中
            return label  # 桶名
    return UNKNOWN_TOKEN  # 兜底


def postal_hash_bucket(postal_code: str, *, n_buckets: int = POSTAL_HASH_BUCKETS) -> str:  # 稳定哈希分桶
    digest = hashlib.sha256(postal_code.encode("utf-8")).hexdigest()  # 稳定
    bucket = int(digest[:8], 16) % n_buckets  # 0..n-1
    return f"h{bucket:02d}"  # 字符串桶


def postal_frequency_bucket(count: int) -> str:  # 频次分桶，不用连续邮编
    if count <= 1:  # 只出现一次
        return "singleton"  # 稀有
    if count <= 5:  # 2-5
        return "rare"  # 少
    if count <= 50:  # 6-50
        return "medium"  # 中
    return "common"  # 常见


def _postal_frequency_map(customers: pd.DataFrame) -> dict[str, int]:  # 全表 postal 频次
    if "postal_code" not in customers.columns:  # 无列
        return {}  # 空
    counts: dict[str, int] = {}  # 计数
    for value in customers["postal_code"]:  # 逐行
        if _is_missing(value):  # 缺失
            continue  # 跳过
        code = str(value).strip()  # 文本
        if not code or code.lower() in {"nan", "<na>", "none"}:  # 空
            continue  # 跳过
        counts[code] = counts.get(code, 0) + 1  # 累加
    return counts  # 返回


def unknown_customer_row(user_id: str) -> dict[str, object]:  # 交易有、主数据无的用户
    return {  # 一行 unknown
        "user_id": canonical_user_id(user_id),  # 用户
        "is_unknown_customer:float": 1.0,  # 补齐行
        "feature_version": CUSTOMER_FEATURE_SCHEMA_VERSION,  # 版本
        "age:float": 0.0,  # 不填均值
        "age_bucket:token": UNKNOWN_TOKEN,  # 未知桶
        "age_missing:float": 1.0,  # 缺失
        "FN:float": 0.0,  # 三态未知
        "FN_missing:float": 1.0,  # 缺失
        "Active:float": 0.0,  # 三态未知
        "Active_missing:float": 1.0,  # 缺失
        "club_member_status:token": UNKNOWN_TOKEN,  # 类别
        "club_member_status_missing:float": 1.0,  # 缺失
        "fashion_news_frequency:token": UNKNOWN_TOKEN,  # 类别
        "fashion_news_frequency_missing:float": 1.0,  # 缺失
        "postal_code:token": UNKNOWN_TOKEN,  # 不做连续值
        "postal_code_hash_bucket:token": UNKNOWN_TOKEN,  # 哈希桶
        "postal_code_freq_bucket:token": UNKNOWN_TOKEN,  # 频次桶
        "postal_code_missing:float": 1.0,  # 缺失
    }  # 结束


def load_customers_table(customers_path: Path) -> pd.DataFrame:  # 读 customers，缺列补空
    customers_path = Path(customers_path)  # 规范化
    if not customers_path.is_file():  # 缺文件
        raise FileNotFoundError(f"customers.csv not found: {customers_path}")  # 无法构建
    frame = pd.read_csv(customers_path, dtype="string")  # 全字符串
    if "customer_id" not in frame.columns:  # 主键
        raise ValueError("customers must contain customer_id")  # 报错
    for col in CUSTOMER_SOURCE_COLUMNS:  # 六列
        if col not in frame.columns:  # 缺列
            frame[col] = pd.NA  # 补空
    frame["user_id"] = frame["customer_id"].map(canonical_user_id)  # 统一 ID
    frame = frame.drop_duplicates("user_id", keep="first")  # 一用户一行
    return frame  # 返回


def build_customer_feature_table(  # 目录用户 + 交互缺失 unknown
    customers: pd.DataFrame,  # 已 load
    *,
    extra_user_ids: set[str] | None = None,  # 切分里出现的主数据外用户
    keep_full_customer_universe: bool = True,  # False 时只留 extra 与目录交集
) -> pd.DataFrame:  # 特征表
    if "user_id" not in customers.columns:  # 需要 user_id
        raise ValueError("customers table must contain user_id")  # 报错
    extra = {canonical_user_id(user_id) for user_id in (extra_user_ids or set())}  # 规范化
    source = customers.copy()  # 不改调用方
    if not keep_full_customer_universe:  # 快速实验
        if extra:  # 有交互用户
            source = source[source["user_id"].isin(extra)].copy()  # 只留出现过的
        else:  # 无交互
            source = source.iloc[0:0].copy()  # 空表
    postal_freq = _postal_frequency_map(source if not source.empty else customers)  # 频次在全表上算
    rows: list[dict[str, object]] = []  # 逐行
    for raw in source.to_dict(orient="records"):  # 目录用户
        user_id = canonical_user_id(raw["user_id"])  # ID
        age, age_bucket, age_missing = parse_age(raw.get("age"))  # 年龄
        fn_value, fn_missing = parse_binary_flag(raw.get("FN"))  # FN 三态
        active_value, active_missing = parse_binary_flag(raw.get("Active"))  # Active 三态
        club_token = clean_category_token(raw.get("club_member_status"))  # 会员状态
        club_missing = 1.0 if club_token == UNKNOWN_TOKEN else 0.0  # 存在标记
        news_token = clean_category_token(raw.get("fashion_news_frequency"))  # 时尚资讯频率
        news_missing = 1.0 if news_token == UNKNOWN_TOKEN else 0.0  # 存在标记
        postal_raw = raw.get("postal_code")  # 邮编
        if _is_missing(postal_raw):  # 缺失
            postal_token = UNKNOWN_TOKEN  # unknown
            hash_bucket = UNKNOWN_TOKEN  # unknown
            freq_bucket = UNKNOWN_TOKEN  # unknown
            postal_missing = 1.0  # 缺失
        else:  # 有邮编
            postal_text = str(postal_raw).strip()  # 文本
            if not postal_text or postal_text.lower() in {"nan", "<na>", "none"}:  # 空
                postal_token = UNKNOWN_TOKEN  # unknown
                hash_bucket = UNKNOWN_TOKEN  # unknown
                freq_bucket = UNKNOWN_TOKEN  # unknown
                postal_missing = 1.0  # 缺失
            else:  # 有效邮编
                postal_token = clean_category_token(postal_text)  # token，不做连续距离
                hash_bucket = postal_hash_bucket(postal_text)  # 哈希桶
                freq_bucket = postal_frequency_bucket(postal_freq.get(postal_text, 1))  # 频次桶
                postal_missing = 0.0  # 不缺失
        rows.append(  # 追加
            {
                "user_id": user_id,  # 用户
                "is_unknown_customer:float": 0.0,  # 有主数据
                "feature_version": CUSTOMER_FEATURE_SCHEMA_VERSION,  # 版本
                "age:float": age,  # 年龄
                "age_bucket:token": age_bucket,  # 分桶
                "age_missing:float": age_missing,  # 缺失
                "FN:float": fn_value,  # 1/0
                "FN_missing:float": fn_missing,  # 三态 missing
                "Active:float": active_value,  # 1/0
                "Active_missing:float": active_missing,  # 三态 missing
                "club_member_status:token": club_token,  # 会员
                "club_member_status_missing:float": club_missing,  # 存在标记
                "fashion_news_frequency:token": news_token,  # 资讯频率
                "fashion_news_frequency_missing:float": news_missing,  # 存在标记
                "postal_code:token": postal_token,  # 类别 token
                "postal_code_hash_bucket:token": hash_bucket,  # 哈希桶
                "postal_code_freq_bucket:token": freq_bucket,  # 频次桶
                "postal_code_missing:float": postal_missing,  # 缺失
            }
        )  # 行结束
    catalog = pd.DataFrame(rows)  # 目录表
    if catalog.empty:  # 空
        catalog = pd.DataFrame(columns=list(unknown_customer_row("u").keys()))  # 保留列
        catalog["user_id"] = catalog["user_id"].astype("string")  # 类型
    present = set(catalog["user_id"].astype(str)) if not catalog.empty else set()  # 已有
    missing_ids = sorted(extra - present)  # 交互有、主数据无
    if missing_ids:  # 必须补 unknown
        unknown_df = pd.DataFrame([unknown_customer_row(user_id) for user_id in missing_ids])  # 补齐
        catalog = pd.concat([catalog, unknown_df], ignore_index=True)  # 合并
    return catalog.sort_values("user_id", kind="mergesort").reset_index(drop=True)  # 稳定排序


def collect_inter_users(inter_paths: list[Path]) -> set[str]:  # 从 hm.*.inter 收集用户
    user_ids: set[str] = set()  # 集合
    for inter_path in inter_paths:  # 每个划分
        path = Path(inter_path)  # 规范化
        if not path.is_file():  # 不存在
            raise FileNotFoundError(f"Missing split file: {path}")  # 报错
        frame = pd.read_csv(path, sep="\t", usecols=["user_id:token"], dtype={"user_id:token": "string"})  # 只读用户
        normalized = frame["user_id:token"].map(canonical_user_id)  # 规范
        user_ids.update(user_id for user_id in normalized if user_id)  # 加入
    return user_ids  # 返回


def write_customer_features_parquet(features: pd.DataFrame, output_path: Path) -> Path:  # 写出 parquet
    output_path = Path(output_path)  # 规范化
    if output_path.suffix == "":  # 目录
        output_path = output_path / "customers.parquet"  # 默认名
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 建目录
    features.to_parquet(output_path, index=False, engine="pyarrow")  # 写出
    return output_path  # 返回


def build_customer_features(  # CLI / 数据准备入口
    customers_path: Path = DEFAULT_CUSTOMERS,  # 用户主数据
    output_path: Path | None = None,  # parquet 路径
    inter_paths: tuple[Path, ...] | None = None,  # 切分交互，决定 extra_user_ids
    keep_full_customer_universe: bool = True,  # 默认保留全量目录
) -> Path:  # 写出路径
    if output_path is None:  # 必须指定
        raise ValueError("output_path is required")  # 报错
    inter_user_ids: set[str] = set()  # 交互用户
    if inter_paths is not None:  # 有切分
        inter_user_ids = collect_inter_users(list(inter_paths))  # 收集
        if not inter_user_ids:  # 空
            raise ValueError("No user ids found in interaction split files.")  # 报错
    customers_df = load_customers_table(customers_path)  # 读表
    features = build_customer_feature_table(  # 构建
        customers_df,  # 主数据
        extra_user_ids=inter_user_ids,  # 补齐 unknown
        keep_full_customer_universe=keep_full_customer_universe,  # 全量目录
    )  # 结束
    written = write_customer_features_parquet(features, output_path)  # 落盘
    missing_metadata = int((features["is_unknown_customer:float"] == 1.0).sum())  # unknown 行
    print(f"saved customer features: {written} ({len(features):,} rows)")  # 路径
    if inter_user_ids:  # 有交互
        print(f"covered user ids from inter: {len(inter_user_ids):,}")  # 交互用户数
    print(f"missing metadata backfilled: {missing_metadata:,}")  # 补齐数
    return written  # 返回

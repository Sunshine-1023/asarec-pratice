"""Stream raw H&M tables and write a reproducible data profile JSON."""  # 流式扫描原始表并写出可复现 profile

from __future__ import annotations  # 延迟注解

import argparse  # CLI
import json  # 写出 JSON
import math  # 价格统计
import re  # 检查被读成浮点的 ID
from datetime import datetime, timezone  # 生成时间
from pathlib import Path  # 路径
from typing import Any, Iterable  # 类型

import pandas as pd  # 分块读取

from fashionrec.industrial.data.manifest import sha256_file  # 复用流式哈希


DEFAULT_TRANSACTIONS = Path("data/raw/transactions_train.csv")  # 默认交易
DEFAULT_CUSTOMERS = Path("data/raw/customers.csv")  # 默认用户
DEFAULT_ARTICLES = Path("data/raw/articles.csv")  # 默认商品
DEFAULT_OUTPUT = Path("outputs/data_profile.json")  # 默认输出
CHUNK_SIZE = 200_000  # 分块行数，避免一次装入 3GB
FLOATISH_ID = re.compile(r"^\d+\.0+$")  # CSV 被 excel/pandas 读成浮点后再转字符串的形态


def _missing_rate(n_missing: int, n_rows: int) -> float:  # 缺失率
    return float(n_missing / n_rows) if n_rows else 0.0  # 空表为 0


def _update_extrema(current: float | None, value: float, op: str) -> float:  # 流式更新最小/最大
    if current is None or (math.isnan(current) and not math.isnan(value)):  # 初值
        return value  # 采用当前值
    if math.isnan(value):  # 跳过 NaN
        return current  # 保持
    return min(current, value) if op == "min" else max(current, value)  # 比较


def profile_transactions(path: Path, chunk_size: int = CHUNK_SIZE) -> dict[str, Any]:  # 流式扫描交易
    n_rows = 0  # 行数
    users: set[str] = set()  # 交易用户
    items: set[str] = set()  # 交易 SKU
    missing: dict[str, int] = {}  # 各列缺失
    n_floatish_user = 0  # 用户 ID 浮点形态
    n_floatish_item = 0  # 商品 ID 浮点形态
    n_price_null = 0  # 价格空
    n_price_non_positive = 0  # 价格非正
    min_price: float | None = None  # 最低价
    max_price: float | None = None  # 最高价
    n_date_null = 0  # 日期空
    min_date: pd.Timestamp | None = None  # 最早日期
    max_date: pd.Timestamp | None = None  # 最晚日期
    n_decreasing_adjacent = 0  # 相邻行日期回退次数（文件未排序时仅作诊断）
    prev_date: pd.Timestamp | None = None  # 上一行日期
    columns: list[str] = []  # schema

    reader = pd.read_csv(  # 分块读取
        path,  # 路径
        dtype={"customer_id": "string", "article_id": "string"},  # ID 强制字符串
        parse_dates=["t_dat"],  # 解析日期
        chunksize=chunk_size,  # 分块
    )  # 读取结束
    for chunk in reader:  # 逐块
        if not columns:  # 记录 schema
            columns = list(chunk.columns)  # 列名
            missing = {col: 0 for col in columns}  # 初始化缺失计数
        n_rows += len(chunk)  # 累加行数
        for col in columns:  # 各列缺失
            missing[col] += int(chunk[col].isna().sum())  # 缺失个数
        user_ids = chunk["customer_id"].astype("string")  # 用户
        item_ids = chunk["article_id"].astype("string")  # 商品
        users.update(uid for uid in user_ids.dropna().tolist() if uid and uid != "<NA>")  # 去重用户
        items.update(iid for iid in item_ids.dropna().tolist() if iid and iid != "<NA>")  # 去重商品
        n_floatish_user += int(user_ids.fillna("").str.match(FLOATISH_ID).sum())  # 浮点用户 ID
        n_floatish_item += int(item_ids.fillna("").str.match(FLOATISH_ID).sum())  # 浮点商品 ID
        if "price" in chunk.columns:  # 有价格列
            prices = pd.to_numeric(chunk["price"], errors="coerce")  # 转数值
            n_price_null += int(prices.isna().sum())  # 空价格
            valid_prices = prices.dropna()  # 有效价格
            n_price_non_positive += int((valid_prices <= 0).sum())  # 非正
            if not valid_prices.empty:  # 有价格
                min_price = _update_extrema(min_price, float(valid_prices.min()), "min")  # 最小
                max_price = _update_extrema(max_price, float(valid_prices.max()), "max")  # 最大
        dates = pd.to_datetime(chunk["t_dat"], errors="coerce")  # 日期
        n_date_null += int(dates.isna().sum())  # 空日期
        valid_dates = dates.dropna()  # 有效日期
        if not valid_dates.empty:  # 有日期
            chunk_min = pd.Timestamp(valid_dates.min())  # 块最小
            chunk_max = pd.Timestamp(valid_dates.max())  # 块最大
            min_date = chunk_min if min_date is None else min(min_date, chunk_min)  # 全局最小
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)  # 全局最大
            ordered = valid_dates.tolist()  # 块内顺序
            if prev_date is not None and ordered and pd.Timestamp(ordered[0]) < prev_date:  # 跨块回退
                n_decreasing_adjacent += 1  # 计数
            n_decreasing_adjacent += int(sum(later < earlier for earlier, later in zip(ordered, ordered[1:])))  # 块内回退
            prev_date = pd.Timestamp(ordered[-1])  # 记住块尾

    return {  # 交易 profile
        "path": str(path),  # 路径
        "exists": True,  # 已存在，否则调用方不会进来
        "n_rows": n_rows,  # 行数
        "n_users": len(users),  # 用户数
        "n_items": len(items),  # SKU 数
        "columns": columns,  # schema
        "missing_rate": {col: _missing_rate(count, n_rows) for col, count in missing.items()},  # 缺失率
        "ids_read_as_string": True,  # 本函数强制 string dtype
        "n_floatish_customer_id": n_floatish_user,  # 疑似浮点用户 ID
        "n_floatish_article_id": n_floatish_item,  # 疑似浮点商品 ID
        "price": {  # 价格诊断
            "n_null": n_price_null,  # 空
            "n_non_positive": n_price_non_positive,  # 非正
            "min": min_price,  # 最小
            "max": max_price,  # 最大
        },  # 价格结束
        "dates": {  # 日期诊断
            "n_null": n_date_null,  # 空
            "min": None if min_date is None else str(min_date.date()),  # 最早
            "max": None if max_date is None else str(max_date.date()),  # 最晚
            "n_decreasing_adjacent": n_decreasing_adjacent,  # 相邻回退
        },  # 日期结束
        "user_ids": users,  # 供 join 覆盖率使用，写出前会删
        "item_ids": items,  # 供 join 覆盖率使用
        "sha256": sha256_file(path),  # 文件哈希
        "nbytes": path.stat().st_size,  # 字节数
    }  # 返回


def profile_customers(path: Path, chunk_size: int = CHUNK_SIZE) -> dict[str, Any]:  # 流式扫描用户表
    n_rows = 0  # 行数
    ids: set[str] = set()  # 用户 ID
    missing: dict[str, int] = {}  # 缺失
    columns: list[str] = []  # schema
    n_floatish = 0  # 浮点 ID
    reader = pd.read_csv(path, dtype={"customer_id": "string"}, chunksize=chunk_size)  # 分块
    for chunk in reader:  # 逐块
        if not columns:  # schema
            columns = list(chunk.columns)  # 列
            missing = {col: 0 for col in columns}  # 缺失计数
        n_rows += len(chunk)  # 行数
        for col in columns:  # 缺失
            missing[col] += int(chunk[col].isna().sum())  # 累加
        series = chunk["customer_id"].astype("string")  # ID
        ids.update(uid for uid in series.dropna().tolist() if uid and uid != "<NA>")  # 去重
        n_floatish += int(series.fillna("").str.match(FLOATISH_ID).sum())  # 浮点形态
    return {  # 用户 profile
        "path": str(path),  # 路径
        "n_rows": n_rows,  # 行数
        "n_ids": len(ids),  # 唯一 ID
        "columns": columns,  # schema
        "missing_rate": {col: _missing_rate(count, n_rows) for col, count in missing.items()},  # 缺失率
        "ids_read_as_string": True,  # 强制字符串
        "n_floatish_customer_id": n_floatish,  # 浮点 ID 数
        "ids": ids,  # join 用
        "sha256": sha256_file(path),  # 哈希
        "nbytes": path.stat().st_size,  # 字节
    }  # 返回


def profile_articles(path: Path, chunk_size: int = CHUNK_SIZE) -> dict[str, Any]:  # 流式扫描商品表
    n_rows = 0  # 行数
    ids: set[str] = set()  # SKU
    missing: dict[str, int] = {}  # 缺失
    columns: list[str] = []  # schema
    n_floatish = 0  # 浮点 ID
    article_to_code: dict[str, str] = {}  # article_id -> 首次见到的 product_code
    n_conflicting_product_code = 0  # 同一 article_id 对应多个 product_code
    n_product_codes = 0  # 下面用 set 计
    product_codes: set[str] = set()  # 款式
    reader = pd.read_csv(path, dtype={"article_id": "string", "product_code": "string"}, chunksize=chunk_size)  # 分块
    for chunk in reader:  # 逐块
        if not columns:  # schema
            columns = list(chunk.columns)  # 列
            missing = {col: 0 for col in columns}  # 缺失
        n_rows += len(chunk)  # 行数
        for col in columns:  # 缺失
            missing[col] += int(chunk[col].isna().sum())  # 累加
        article_ids = chunk["article_id"].astype("string")  # SKU
        ids.update(iid for iid in article_ids.dropna().tolist() if iid and iid != "<NA>")  # 去重
        n_floatish += int(article_ids.fillna("").str.match(FLOATISH_ID).sum())  # 浮点
        if "product_code" in chunk.columns:  # 有款式列
            for article_id, product_code in zip(  # 检查一对多
                article_ids.tolist(), chunk["product_code"].astype("string").tolist()  # 两列
            ):  # 逐行
                if not article_id or article_id == "<NA>":  # 空 SKU
                    continue  # 跳过
                code = "" if product_code in {None, "<NA>"} else str(product_code)  # 款式
                product_codes.add(code)  # 记录款式
                previous = article_to_code.get(article_id)  # 已见映射
                if previous is None:  # 首次
                    article_to_code[article_id] = code  # 记下
                elif previous != code:  # 同一 SKU 多个款式，不合法
                    n_conflicting_product_code += 1  # 计数
    n_product_codes = len(product_codes)  # 款式数
    return {  # 商品 profile
        "path": str(path),  # 路径
        "n_rows": n_rows,  # 行数
        "n_ids": len(ids),  # SKU 数
        "n_product_codes": n_product_codes,  # 款式数
        "n_columns": len(columns),  # 字段数
        "columns": columns,  # schema
        "missing_rate": {col: _missing_rate(count, n_rows) for col, count in missing.items()},  # 缺失率
        "ids_read_as_string": True,  # 强制字符串
        "n_floatish_article_id": n_floatish,  # 浮点 ID
        "article_id_to_product_code": {  # 一对多检查
            "n_article_ids_with_conflicting_product_code": n_conflicting_product_code,  # 冲突数
            "mapping_is_many_skus_to_one_style": True,  # 合法方向是多 SKU 对一款式
        },  # 映射结束
        "ids": ids,  # join 用
        "sha256": sha256_file(path),  # 哈希
        "nbytes": path.stat().st_size,  # 字节
    }  # 返回


def _coverage(left: Iterable[str], right: set[str]) -> dict[str, Any]:  # 左集合对右集合的覆盖
    left_set = set(left)  # 拷贝
    n_left = len(left_set)  # 左大小
    n_missing = sum(1 for item in left_set if item not in right)  # 未覆盖
    return {  # 覆盖率
        "n_left": n_left,  # 左
        "n_missing": n_missing,  # 缺失
        "coverage": 1.0 if n_left == 0 else float((n_left - n_missing) / n_left),  # 比例
    }  # 返回


def build_data_profile(  # 组装三表 profile
    transactions_path: Path,  # 交易
    customers_path: Path,  # 用户
    articles_path: Path,  # 商品
    chunk_size: int = CHUNK_SIZE,  # 分块
) -> dict[str, Any]:  # JSON 载荷
    for path, name in (  # 缺文件立即失败
        (transactions_path, "transactions"),  # 交易
        (customers_path, "customers"),  # 用户
        (articles_path, "articles"),  # 商品
    ):  # 检查结束
        if not path.exists():  # 不存在
            raise FileNotFoundError(f"{name} file not found: {path}")  # 明确报错

    transactions = profile_transactions(transactions_path, chunk_size=chunk_size)  # 交易
    customers = profile_customers(customers_path, chunk_size=chunk_size)  # 用户
    articles = profile_articles(articles_path, chunk_size=chunk_size)  # 商品
    txn_users = transactions.pop("user_ids")  # 取出 join 集合
    txn_items = transactions.pop("item_ids")  # SKU
    customer_ids = customers.pop("ids")  # 用户主数据
    article_ids = articles.pop("ids")  # 商品主数据
    return {  # 完整 profile
        "generated_at": datetime.now(timezone.utc).isoformat(),  # 生成时间
        "chunk_size": chunk_size,  # 分块大小
        "files": {  # 分表
            "transactions": transactions,  # 交易
            "customers": customers,  # 用户
            "articles": articles,  # 商品
        },  # 分表结束
        "join_coverage": {  # 关联覆盖
            "txn_users_in_customers": _coverage(txn_users, customer_ids),  # 交易用户是否都在 customers
            "txn_items_in_articles": _coverage(txn_items, article_ids),  # 交易 SKU 是否都在 articles
        },  # 覆盖结束
    }  # 返回


def write_data_profile(payload: dict[str, Any], output_path: Path) -> Path:  # 写 JSON
    output_path = Path(output_path)  # 路径
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 目录
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 写出
    return output_path  # 返回


def main(argv: list[str] | None = None) -> None:  # CLI
    parser = argparse.ArgumentParser(  # 参数
        prog="fashionrec profile-data",  # 子命令名
        description="Stream raw H&M CSVs and write a reproducible schema/quality profile.",  # 说明
    )  # 解析器结束
    parser.add_argument("--transactions", type=Path, default=DEFAULT_TRANSACTIONS)  # 交易路径
    parser.add_argument("--customers", type=Path, default=DEFAULT_CUSTOMERS)  # 用户路径
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)  # 商品路径
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)  # 输出
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)  # 分块
    args = parser.parse_args(argv)  # 解析
    payload = build_data_profile(  # 扫描
        args.transactions,  # 交易
        args.customers,  # 用户
        args.articles,  # 商品
        chunk_size=args.chunk_size,  # 分块
    )  # 扫描结束
    out = write_data_profile(payload, args.output)  # 写出
    print(f"Wrote data profile: {out}")  # 提示
    print(json.dumps(  # 打印摘要，不含超大集合
        {  # 摘要
            "transactions_rows": payload["files"]["transactions"]["n_rows"],  # 交易行
            "customers_rows": payload["files"]["customers"]["n_rows"],  # 用户行
            "articles_rows": payload["files"]["articles"]["n_rows"],  # 商品行
            "join_coverage": payload["join_coverage"],  # 覆盖
        },  # 摘要结束
        ensure_ascii=False,  # 中文
        indent=2,  # 缩进
    ))  # 打印结束


if __name__ == "__main__":  # 直接运行
    main()  # 入口

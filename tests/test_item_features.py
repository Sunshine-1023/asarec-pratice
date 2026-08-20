"""Tests for SKU and product_code item features."""  # 2.1 商品静态特征

from __future__ import annotations  # 延迟注解

from pathlib import Path  # 路径

import pandas as pd  # 表格

from fashionrec.data.build_item_features import build_item_features  # RecBole 写出
from fashionrec.data.command import processed_layout  # 布局
from fashionrec.data.item_features import (  # 特征
    ARTICLE_SOURCE_COLUMNS,  # 24 字段
    ITEM_FEATURE_SCHEMA_VERSION,  # 版本
    RECB_ITEM_FILE_COLUMNS,  # RecBole 列
    build_item_feature_table,  # 纯函数
    load_articles_table,  # 读 CSV
    recbole_item_frame,  # 切片
    write_item_features_parquet,  # parquet
)


def _articles_csv(path: Path, rows: list[dict[str, object]]) -> Path:  # 写带齐 24 字段的 articles
    frame = pd.DataFrame(rows)  # 行
    for col in ("article_id", *ARTICLE_SOURCE_COLUMNS):  # 缺列补空
        if col not in frame.columns:  # 没有
            frame[col] = pd.NA  # 空
    frame = frame.loc[:, ["article_id", *ARTICLE_SOURCE_COLUMNS]]  # 固定顺序
    frame.to_csv(path, index=False)  # 写出
    return path  # 返回


def _write_inter(path: Path, item_ids: list[str]) -> None:  # 最小 RecBole 交互
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]  # 表头
    for index, item_id in enumerate(item_ids):  # 每件一行
        lines.append(f"u1\t{item_id}\t{index + 1}.0")  # 占位时间
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 写出


def test_numeric_ids_keep_token_and_float_and_style_variant_count(tmp_path: Path) -> None:  # 款式层级
    source = _articles_csv(
        tmp_path / "articles.csv",
        [
            {  # 款 108775 红色
                "article_id": "0108775015",
                "product_code": "108775",
                "prod_name": "Strap top",
                "product_type_no": "253",
                "product_type_name": "Vest top",
                "product_group_name": "Garment Upper body",
                "graphical_appearance_no": "1010016",
                "graphical_appearance_name": "Solid",
                "colour_group_code": "9",
                "colour_group_name": "Black",
                "perceived_colour_value_id": "4",
                "perceived_colour_value_name": "Dark",
                "perceived_colour_master_id": "5",
                "perceived_colour_master_name": "Black",
                "department_no": "1676",
                "department_name": "Jersey Basic",
                "index_code": "A",
                "index_name": "Ladieswear",
                "index_group_no": "1",
                "index_group_name": "Ladieswear",
                "section_no": "16",
                "section_name": "Womens Everyday Basics",
                "garment_group_no": "1002",
                "garment_group_name": "Jersey Basic",
                "detail_desc": "Jersey top with narrow straps.",
            },
            {  # 同款白色，变体数应为 2
                "article_id": "0108775044",
                "product_code": "108775",
                "prod_name": "Strap top",
                "colour_group_code": "10",
                "colour_group_name": "White",
                "perceived_colour_value_name": "Light",
                "perceived_colour_master_name": "White",
                "detail_desc": "Jersey top with narrow straps.",
            },
            {  # 另一款，目录长尾，即使不在交互里也要留下
                "article_id": "0111609001",
                "product_code": "111609",
                "prod_name": "Other style",
                "colour_group_name": "Blue",
            },
        ],
    )
    articles = load_articles_table(source)
    features = build_item_feature_table(articles, extra_item_ids={"0108775015"}, keep_full_item_universe=True)
    assert set(features["item_id"]) == {"0108775015", "0108775044", "0111609001"}  # 不截断目录
    red = features.set_index("item_id").loc["0108775015"]
    assert red["product_code:token"] == "108775"
    assert float(red["product_code:float"]) == 108775.0
    assert float(red["product_code_missing:float"]) == 0.0
    assert float(red["n_style_variants:float"]) == 2.0  # 同款两 SKU
    assert red["prod_name:token"] == "strap top"
    assert float(red["prod_name_n_tokens:float"]) == 2.0
    assert float(red["prod_name_missing:float"]) == 0.0
    assert float(red["detail_desc_missing:float"]) == 0.0
    assert red["colour_combo:token"] == "Black__Black"
    assert red["pattern_combo:token"] == "Solid"
    assert red["shade_combo:token"] == "Dark"
    assert red["material_combo:token"] == "Garment_Upper_body__Jersey_Basic"
    assert red["feature_version"] == ITEM_FEATURE_SCHEMA_VERSION
    other = features.set_index("item_id").loc["0111609001"]
    assert float(other["n_style_variants:float"]) == 1.0  # 单独一款


def test_missing_metadata_is_unknown_row_not_dropped() -> None:  # 未见商品必须保留
    articles = pd.DataFrame(
        {
            "item_id": ["0108775015"],
            "article_id": ["0108775015"],
            **{col: [pd.NA] for col in ARTICLE_SOURCE_COLUMNS},
        }
    )
    articles["product_code"] = ["108775"]
    features = build_item_feature_table(
        articles,
        extra_item_ids={"0108775015", "0999999999"},  # 交互里多一个没有主数据的 SKU
        keep_full_item_universe=True,
    )
    by_id = features.set_index("item_id")
    assert "0999999999" in by_id.index  # 不静默删除
    unknown = by_id.loc["0999999999"]
    assert float(unknown["is_unknown_item:float"]) == 1.0
    assert unknown["product_type_name:token"] == "unknown"
    assert float(unknown["detail_desc_missing:float"]) == 1.0
    assert float(unknown["n_style_variants:float"]) == 0.0


def test_empty_text_is_missing_not_filled_with_mean() -> None:  # 文本缺失打标记
    articles = pd.DataFrame(
        {
            "item_id": ["0000000001"],
            "article_id": ["0000000001"],
            **{col: [pd.NA] for col in ARTICLE_SOURCE_COLUMNS},
        }
    )
    features = build_item_feature_table(articles)
    row = features.iloc[0]
    assert row["prod_name:token"] == "unknown"
    assert float(row["prod_name_n_chars:float"]) == 0.0
    assert float(row["prod_name_missing:float"]) == 1.0
    assert float(row["detail_desc_missing:float"]) == 1.0


def test_keep_full_item_universe_false_still_backfills_unknown() -> None:  # 快速实验可丢掉未出现目录 SKU
    articles = pd.DataFrame(
        {
            "item_id": ["0000000001", "0000000002"],
            "article_id": ["0000000001", "0000000002"],
            **{col: ["a", "b"] for col in ARTICLE_SOURCE_COLUMNS},
        }
    )
    features = build_item_feature_table(
        articles,
        extra_item_ids={"0000000001", "0000000009"},
        keep_full_item_universe=False,
    )
    assert set(features["item_id"]) == {"0000000001", "0000000009"}  # 2 不在交互里被去掉
    assert float(features.set_index("item_id").loc["0000000009", "is_unknown_item:float"]) == 1.0


def test_build_item_features_writes_recbole_slice_and_full_parquet(tmp_path: Path) -> None:  # 双写
    articles = _articles_csv(
        tmp_path / "articles.csv",
        [
            {"article_id": "0108775015", "product_code": "108775", "product_type_name": "Vest top"},
            {"article_id": "0111609001", "product_code": "111609", "product_type_name": "Sweater"},  # 不在序列里
        ],
    )
    train = tmp_path / "train.inter"
    valid = tmp_path / "valid.inter"
    test = tmp_path / "test.inter"
    _write_inter(train, ["0108775015", "0999999999"])  # 含未见 metadata
    _write_inter(valid, ["0108775015"])
    _write_inter(test, ["0108775015"])
    recb_path = tmp_path / "hm_seq.item"
    parquet_path = tmp_path / "item_features" / "items.parquet"
    build_item_features(
        articles_path=articles,
        output_path=recb_path,
        inter_paths=(train, valid, test),
        features_output_path=parquet_path,
        keep_full_item_universe=True,
    )
    recb = pd.read_csv(recb_path, sep="\t", dtype="string")
    assert list(recb.columns) == list(RECB_ITEM_FILE_COLUMNS)  # SASRecF 列不变
    assert set(recb["item_id:token"]) == {"0108775015", "0999999999"}  # 序列对齐，含 unknown
    assert "0111609001" not in set(recb["item_id:token"])  # RecBole 不灌未出现目录 SKU
    unknown = recb[recb["item_id:token"] == "0999999999"].iloc[0]
    assert unknown["product_type_name:token"] == "unknown"
    features = pd.read_parquet(parquet_path)
    assert set(features["item_id"]) == {"0108775015", "0111609001", "0999999999"}  # parquet 保留全量目录
    assert features["feature_version"].unique().tolist() == [ITEM_FEATURE_SCHEMA_VERSION]


def test_recbole_item_frame_covers_requested_ids() -> None:  # 切片含 unknown
    articles = pd.DataFrame(
        {
            "item_id": ["0000000001"],
            "article_id": ["0000000001"],
            **{col: ["x"] for col in ARTICLE_SOURCE_COLUMNS},
        }
    )
    features = build_item_feature_table(articles, extra_item_ids={"0000000001", "0000000002"})
    recb = recbole_item_frame(features, item_ids={"0000000002"})
    assert list(recb["item_id:token"]) == ["0000000002"]
    assert recb.iloc[0]["product_type_name:token"] == "unknown"


def test_write_item_features_parquet_roundtrip(tmp_path: Path) -> None:  # 落盘
    articles = pd.DataFrame(
        {
            "item_id": ["0000000001"],
            "article_id": ["0000000001"],
            **{col: ["hello world"] for col in ARTICLE_SOURCE_COLUMNS},
        }
    )
    features = build_item_feature_table(articles)
    written = write_item_features_parquet(features, tmp_path / "item_features")  # 目录也能写
    loaded = pd.read_parquet(written)
    assert loaded.iloc[0]["item_id"] == "0000000001"
    assert written.name == "items.parquet"


def test_processed_layout_includes_item_features(tmp_path: Path) -> None:  # 布局
    assert processed_layout(tmp_path)["item_features"] == tmp_path / "item_features" / "items.parquet"

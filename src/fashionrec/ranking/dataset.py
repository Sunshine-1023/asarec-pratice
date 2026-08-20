"""Assemble LambdaRank rows from fixed candidates, PIT features, and next-basket labels."""  # 排序训练表

from __future__ import annotations  # 延迟注解

from collections.abc import Iterable, Mapping  # 类型
from dataclasses import dataclass, field  # 产物
from pathlib import Path  # 落盘

import pandas as pd  # 表

from fashionrec.data.item_features import UNKNOWN_TOKEN  # 缺失 token 默认值
from fashionrec.domain.candidates import Candidate  # 候选契约
from fashionrec.domain.ids import canonical_item_id, canonical_user_id  # ID
from fashionrec.ranking.features import (  # 通道证据
    build_ranking_features,
    lambda_rank_group_sizes,
    ranking_group_id,
)


RANKING_DATASET_SCHEMA_VERSION = "hm.ranking_dataset.v1"  # 训练表语义
KEY_COLUMNS = ("user_id", "item_id", "snapshot_date", "group_id", "split")  # 主键
LABEL_COLUMNS = ("label", "relevance")  # 二值相关性；后续可改为分级
META_DROP_COLUMNS = ("feature_version", "split", "as_of_date")  # 特征表里不并入的元数据
FEATURE_PREFIXES = {  # 避免列名碰撞
    "user": "user__",
    "customer": "customer__",
    "item": "item__",
    "cross": "cross__",
}


@dataclass(frozen=True, slots=True)
class RankingDataset:
    frame: pd.DataFrame  # 一行一个 user-item-snapshot
    missing_rates: dict[str, float] = field(default_factory=dict)  # 各特征源未命中比例
    n_uncovered_labels: int = 0  # 候选集外的未来购买，不得当正例
    schema_version: str = RANKING_DATASET_SCHEMA_VERSION  # 版本

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    @property
    def n_positives(self) -> int:
        if "label" not in self.frame.columns or self.frame.empty:
            return 0
        return int(self.frame["label"].sum())

    @property
    def group_sizes(self) -> list[int]:
        return lambda_rank_group_sizes(self.frame)


def _as_day(value: pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _normalize_id_frame(frame: pd.DataFrame, *, user: bool = False, item: bool = False, as_of: bool = False) -> pd.DataFrame:
    out = frame.copy()
    if user and "user_id" in out.columns:
        out["user_id"] = out["user_id"].map(canonical_user_id)
    if item and "item_id" in out.columns:
        out["item_id"] = out["item_id"].map(canonical_item_id)
    if as_of and "as_of_date" in out.columns:
        out["as_of_date"] = pd.to_datetime(out["as_of_date"]).dt.normalize()
    return out


def _positive_label_keys(labels: pd.DataFrame) -> set[tuple[str, str, pd.Timestamp | None]]:
    frame = _normalize_id_frame(labels, user=True, item=True, as_of="as_of_date" in labels.columns)
    if "label_purchase" in frame.columns:
        frame = frame[frame["label_purchase"].fillna(0).astype(int) == 1]
    elif "label" in frame.columns:
        frame = frame[frame["label"].fillna(0).astype(int) == 1]
    keys: set[tuple[str, str, pd.Timestamp | None]] = set()
    has_as_of = "as_of_date" in frame.columns
    for row in frame.itertuples(index=False):
        as_of = _as_day(row.as_of_date) if has_as_of else None
        keys.add((str(row.user_id), str(row.item_id), as_of))
    return keys


def _attach_labels(frame: pd.DataFrame, labels: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    if labels is None:
        return frame, 0
    if frame.empty:
        return frame.assign(label=pd.Series(dtype="int64"), relevance=pd.Series(dtype="int64")), 0
    positive = _positive_label_keys(labels)
    candidate_keys: set[tuple[str, str, pd.Timestamp | None]] = set()
    label_values: list[int] = []
    for row in frame.itertuples(index=False):
        snapshot = _as_day(row.snapshot_date)
        keyed = (str(row.user_id), str(row.item_id), snapshot)
        unkeyed = (str(row.user_id), str(row.item_id), None)
        candidate_keys.add(keyed)
        hit = keyed in positive or unkeyed in positive
        label_values.append(int(hit))
    dated_positives = {key for key in positive if key[2] is not None}
    uncovered = len(dated_positives - candidate_keys) if dated_positives else len(positive - {(u, i, None) for u, i, _ in candidate_keys})
    out = frame.copy()
    out["label"] = label_values
    out["relevance"] = label_values  # 首版 purchase=1，非购买候选=0
    return out, uncovered


def _feature_value_columns(frame: pd.DataFrame, keys: tuple[str, ...]) -> list[str]:
    drop = set(keys) | set(META_DROP_COLUMNS)
    return [col for col in frame.columns if col not in drop]


def _default_for_column(name: str, dtype: object) -> object:
    text = str(name)
    if text.endswith(":token") or pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return UNKNOWN_TOKEN
    return 0.0


def _join_feature_table(
    frame: pd.DataFrame,
    features: pd.DataFrame | None,
    *,
    source: str,
    keys: tuple[str, ...],
    feature_as_of: bool = False,
) -> tuple[pd.DataFrame, float]:
    missing_col = f"{source}_features_missing:float"
    if features is None:
        out = frame.copy()
        out[missing_col] = 1.0
        return out, 1.0
    prepared = _normalize_id_frame(
        features,
        user="user_id" in keys,
        item="item_id" in keys,
        as_of=feature_as_of,
    )
    if feature_as_of and "as_of_date" in prepared.columns:
        prepared = prepared.rename(columns={"as_of_date": "snapshot_date"})
        join_keys = tuple("snapshot_date" if key == "as_of_date" else key for key in keys)
    else:
        join_keys = keys
    value_cols = _feature_value_columns(prepared, join_keys)
    prefix = FEATURE_PREFIXES[source]
    renamed = {col: f"{prefix}{col}" for col in value_cols}
    slim = prepared.loc[:, list(join_keys) + value_cols].drop_duplicates(list(join_keys), keep="first")
    slim = slim.rename(columns=renamed)
    merged = frame.merge(slim, on=list(join_keys), how="left", indicator="_join_status")
    matched = merged["_join_status"].eq("both")
    merged = merged.drop(columns="_join_status")
    missing_rate = float((~matched).mean()) if len(merged) else 1.0
    merged[missing_col] = (~matched).astype(float)
    for original, prefixed in renamed.items():
        default = _default_for_column(original, slim[prefixed].dtype if prefixed in slim.columns else float)
        values = merged[prefixed].copy()
        values.loc[~matched] = default
        merged[prefixed] = values
    return merged, missing_rate


def build_ranking_dataset(
    candidates: Iterable[Candidate],
    *,
    channels: Iterable[str],
    snapshot_dates: pd.Timestamp | str | Mapping[str, pd.Timestamp | str] | Mapping[tuple[str, str], pd.Timestamp | str],
    history_lengths: Mapping[str, int] | None = None,
    labels: pd.DataFrame | None = None,
    user_features: pd.DataFrame | None = None,
    customer_features: pd.DataFrame | None = None,
    item_features: pd.DataFrame | None = None,
    cross_features: pd.DataFrame | None = None,
) -> RankingDataset:
    """Build one ranking row per candidate user-item; never add purchases outside the candidate set."""
    candidate_list = list(candidates)
    lengths = dict(history_lengths or {})
    frame = build_ranking_features(
        candidate_list,
        history_lengths=lengths,
        channels=channels,
        snapshot_dates=snapshot_dates,
    )
    if frame.empty:
        empty = pd.DataFrame(columns=[*KEY_COLUMNS, "history_len", "channel_count", *LABEL_COLUMNS])
        return RankingDataset(frame=empty, missing_rates={}, n_uncovered_labels=0)

    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
    frame["group_id"] = [
        ranking_group_id(user_id, snapshot)
        for user_id, snapshot in zip(frame["user_id"], frame["snapshot_date"])
    ]
    frame, uncovered = _attach_labels(frame, labels)

    missing_rates: dict[str, float] = {}
    frame, missing_rates["user"] = _join_feature_table(
        frame, user_features, source="user", keys=("user_id", "as_of_date"), feature_as_of=True
    )
    frame, missing_rates["customer"] = _join_feature_table(
        frame, customer_features, source="customer", keys=("user_id",)
    )
    frame, missing_rates["item"] = _join_feature_table(
        frame, item_features, source="item", keys=("item_id",)
    )
    frame, missing_rates["cross"] = _join_feature_table(
        frame, cross_features, source="cross", keys=("user_id", "item_id", "as_of_date"), feature_as_of=True
    )

    ordered = [col for col in KEY_COLUMNS if col in frame.columns]
    ordered.extend(col for col in ("history_len", "channel_count", "best_channel_rank", "max_channel_score") if col in frame.columns)
    ordered.extend(col for col in LABEL_COLUMNS if col in frame.columns)
    ordered.extend(col for col in frame.columns if col not in ordered)
    frame = frame.loc[:, ordered].sort_values(["group_id", "item_id"], kind="mergesort").reset_index(drop=True)
    return RankingDataset(frame=frame, missing_rates=missing_rates, n_uncovered_labels=uncovered)


def write_ranking_dataset(dataset: RankingDataset, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(path, index=False, engine="pyarrow")
    return path

"""Score ranking tables with a saved LightGBM LambdaRank artifact."""  # 学习排序推理

from __future__ import annotations  # 延迟注解

import argparse  # CLI
import json  # 缺失率
from dataclasses import dataclass  # 排序器
from pathlib import Path  # 路径

import pandas as pd  # 表

from fashionrec.domain.ids import canonical_item_id, canonical_user_id  # ID
from fashionrec.ranking.base import RankedItem  # 统一排序结果
from fashionrec.ranking.train import (  # 与训练共享契约
    RankerSchema,
    load_ranker_booster,
    load_ranker_schema,
    prepare_rank_matrix,
)


@dataclass(slots=True)
class LightGBMRanker:
    booster: object  # LightGBM Booster
    schema: RankerSchema  # 冻结特征
    feature_frame: pd.DataFrame | None = None  # 可选：按用户切表实现 Ranker.rank
    name: str = "lightgbm_lambdarank"

    def bind_features(self, frame: pd.DataFrame) -> "LightGBMRanker":
        self.feature_frame = frame
        return self

    def predict_scores(self, frame: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
        features, _groups, _labels, missing_rates = prepare_rank_matrix(frame, self.schema)
        scores = self.booster.predict(features.to_numpy(), num_iteration=self.schema.best_iteration or -1)  # type: ignore[attr-defined]
        ordered = frame.sort_values(
            [self.schema.group_column, "item_id"] if "item_id" in frame.columns else [self.schema.group_column],
            kind="mergesort",
        )
        return pd.Series(scores, index=ordered.index, name="score"), missing_rates

    def rank(
        self,
        *,
        user_id: str,
        user_history: set[str],
        channel_candidates: dict[str, list[tuple[str, float]]],
        top_k: int,
    ) -> list[RankedItem]:
        _ = channel_candidates
        if self.feature_frame is None:
            raise RuntimeError("LightGBMRanker.rank requires bind_features(frame); use rank_feature_frame for table inference")
        user_key = canonical_user_id(user_id)
        subset = self.feature_frame[self.feature_frame["user_id"].map(canonical_user_id) == user_key]
        if subset.empty:
            return []
        ranked = rank_feature_frame(subset, self, top_k=top_k)
        history = {canonical_item_id(item) for item in user_history}
        items = [
            RankedItem(canonical_item_id(item_id), float(score), int(rank))
            for item_id, score, rank in zip(ranked["item_id"], ranked["score"], ranked["rank"])
            if canonical_item_id(item_id) not in history
        ]
        return [RankedItem(item.item_id, item.score, rank) for rank, item in enumerate(items[:top_k], start=1)]


def load_ranker(model_dir: str | Path) -> LightGBMRanker:
    directory = Path(model_dir)
    schema = load_ranker_schema(directory / "feature_schema.json")
    booster = load_ranker_booster(directory / "model.txt")
    return LightGBMRanker(booster=booster, schema=schema)


def rank_feature_frame(
    frame: pd.DataFrame,
    ranker: LightGBMRanker,
    *,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Add score/rank within each group_id; optionally keep Top-K per group."""
    if frame.empty:
        return frame.copy()
    scores, missing_rates = ranker.predict_scores(frame)
    ranked = frame.loc[scores.index].copy()
    ranked["score"] = scores.to_numpy()
    ranked["_missing_rate_mean"] = float(sum(missing_rates.values()) / len(missing_rates)) if missing_rates else 0.0
    group_col = ranker.schema.group_column
    ranked["rank"] = (
        ranked.groupby(group_col, sort=False)["score"].rank(method="first", ascending=False).astype(int)
    )
    ranked = ranked.sort_values([group_col, "rank", "item_id"] if "item_id" in ranked.columns else [group_col, "rank"], kind="mergesort")
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        ranked = ranked[ranked["rank"] <= top_k]
    ranked.attrs["missing_rates"] = missing_rates
    return ranked.reset_index(drop=True)


def write_scored_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [col for col in ("user_id", "item_id", "snapshot_date", "group_id", "split", "score", "rank", "label", "relevance") if col in frame.columns]
    extra = [col for col in frame.columns if col not in columns and not str(col).startswith("_")]
    frame.loc[:, columns + extra].to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fashionrec ranker-predict", description="Score a ranking parquet with a saved LambdaRank model")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.input_parquet.is_file():
        raise FileNotFoundError(f"input parquet not found: {args.input_parquet}")
    ranker = load_ranker(args.model_dir)
    frame = pd.read_parquet(args.input_parquet)
    ranked = rank_feature_frame(frame, ranker, top_k=args.top_k)
    write_scored_csv(ranked, args.output_csv)
    missing = ranked.attrs.get("missing_rates", {})
    print(json.dumps({"output_csv": str(args.output_csv), "n_rows": int(len(ranked)), "missing_rates": missing}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train LightGBM LambdaRank on a frozen ranking table."""  # 学习排序训练

from __future__ import annotations  # 延迟注解

import argparse  # CLI
import json  # schema / 指标
from dataclasses import asdict, dataclass  # 产物
from pathlib import Path  # 路径
from typing import Any  # booster

import pandas as pd  # 表

from fashionrec.ranking.dataset import KEY_COLUMNS, LABEL_COLUMNS  # 非特征列
from fashionrec.ranking.features import lambda_rank_group_sizes  # group 长度


RANKER_SCHEMA_VERSION = "hm.ranker.v1"  # 推理契约
DEFAULT_N_ESTIMATORS = 200  # 计划建议 100～500
DEFAULT_EARLY_STOPPING = 20  # valid 早停
DEFAULT_LEARNING_RATE = 0.05  # 保守学习率
NON_FEATURE_COLUMNS = set(KEY_COLUMNS) | set(LABEL_COLUMNS) | {"score", "pred", "rank"}  # 不得进模型


@dataclass(frozen=True, slots=True)
class RankerSchema:
    feature_columns: tuple[str, ...]  # 训练时冻结的列序
    defaults: dict[str, float]  # 缺列 / NaN 填充
    label_column: str = "relevance"  # 相关性
    group_column: str = "group_id"  # user-snapshot
    library: str = "lightgbm"  # 实现库
    objective: str = "lambdarank"  # 目标
    schema_version: str = RANKER_SCHEMA_VERSION  # 版本
    best_iteration: int = 0  # 早停迭代
    n_estimators: int = DEFAULT_N_ESTIMATORS  # 配置树数

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_columns"] = list(self.feature_columns)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "RankerSchema":
        columns = tuple(str(col) for col in payload["feature_columns"])
        defaults = {str(key): float(value) for key, value in dict(payload.get("defaults", {})).items()}
        return cls(
            feature_columns=columns,
            defaults=defaults,
            label_column=str(payload.get("label_column", "relevance")),
            group_column=str(payload.get("group_column", "group_id")),
            library=str(payload.get("library", "lightgbm")),
            objective=str(payload.get("objective", "lambdarank")),
            schema_version=str(payload.get("schema_version", RANKER_SCHEMA_VERSION)),
            best_iteration=int(payload.get("best_iteration", 0)),
            n_estimators=int(payload.get("n_estimators", DEFAULT_N_ESTIMATORS)),
        )


@dataclass(frozen=True, slots=True)
class RankerArtifact:
    model_path: Path  # LightGBM txt
    schema_path: Path  # feature schema
    metrics_path: Path  # 训练摘要
    schema: RankerSchema  # 推理契约
    metrics: dict[str, Any]  # 含缺失率 / 早停


def require_lightgbm():  # 延迟导入，避免无依赖环境无法加载模块
    try:
        import lightgbm as lgb  # 官方 LambdaRank
    except ImportError as exc:  # 未安装
        raise RuntimeError("LightGBM is required for LambdaRank; install with: pip install lightgbm") from exc
    except OSError as exc:  # macOS 常见：缺 libomp
        raise RuntimeError("LightGBM found but failed to load (often missing libomp). On macOS: brew install libomp") from exc
    return lgb


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(str(column))
    if not columns:
        raise ValueError("ranking table has no numeric feature columns")
    return columns


def _label_column(frame: pd.DataFrame) -> str:
    if "relevance" in frame.columns:
        return "relevance"
    if "label" in frame.columns:
        return "label"
    raise KeyError("ranking table must contain relevance or label")


def _filter_split(frame: pd.DataFrame, split: str | None) -> pd.DataFrame:
    if split is None or "split" not in frame.columns:
        return frame.copy()
    return frame[frame["split"].astype(str).str.lower() == split].copy()


def _drop_tiny_groups(frame: pd.DataFrame, group_col: str, *, min_size: int = 2) -> pd.DataFrame:
    sizes = frame.groupby(group_col, sort=False).transform("size")
    return frame.loc[sizes >= min_size].copy()


def prepare_rank_matrix(
    frame: pd.DataFrame,
    schema: RankerSchema,
) -> tuple[pd.DataFrame, list[int], pd.Series, dict[str, float]]:
    """Align columns to the frozen schema; record per-column missing rates."""
    if schema.group_column not in frame.columns:
        raise KeyError(f"ranking table missing {schema.group_column}")
    ordered = frame.sort_values([schema.group_column, "item_id"] if "item_id" in frame.columns else [schema.group_column], kind="mergesort")
    features = pd.DataFrame(index=ordered.index)
    missing_rates: dict[str, float] = {}
    for column in schema.feature_columns:
        default = float(schema.defaults.get(column, 0.0))
        if column not in ordered.columns:
            features[column] = default
            missing_rates[column] = 1.0
            continue
        series = pd.to_numeric(ordered[column], errors="coerce")
        missing = series.isna()
        missing_rates[column] = float(missing.mean()) if len(series) else 1.0
        features[column] = series.fillna(default)
    group_sizes = lambda_rank_group_sizes(ordered.rename(columns={schema.group_column: "group_id"}) if schema.group_column != "group_id" else ordered)
    labels = ordered[schema.label_column] if schema.label_column in ordered.columns else pd.Series(dtype="float64")
    return features, group_sizes, labels, missing_rates


def train_lambdarank(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None = None,
    *,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 2026,
    train_split: str = "train",
    valid_split: str = "valid",
) -> tuple[Any, RankerSchema, dict[str, Any]]:
    """Fit LambdaRank on train groups; early-stop on valid only."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be >= 1")
    lgb = require_lightgbm()
    train = _filter_split(train_frame, train_split if "split" in train_frame.columns else None)
    if train.empty:
        raise ValueError("train ranking table is empty after split filter")
    label_col = _label_column(train)
    group_col = "group_id" if "group_id" in train.columns else "user_id"
    train = _drop_tiny_groups(train, group_col)
    if train.empty:
        raise ValueError("train ranking table has no groups with at least 2 items")
    feature_columns = select_feature_columns(train)
    schema = RankerSchema(
        feature_columns=tuple(feature_columns),
        defaults={column: 0.0 for column in feature_columns},
        label_column=label_col,
        group_column=group_col,
        n_estimators=n_estimators,
    )
    x_train, group_train, y_train, train_missing = prepare_rank_matrix(train, schema)
    eval_set = None
    eval_group = None
    valid_missing: dict[str, float] = {}
    if valid_frame is not None and not valid_frame.empty:
        valid = _filter_split(valid_frame, valid_split if "split" in valid_frame.columns else None)
        valid = _drop_tiny_groups(valid, group_col) if not valid.empty else valid
        if not valid.empty:
            if label_col not in valid.columns and "label" in valid.columns:
                valid = valid.rename(columns={"label": label_col})
            x_valid, group_valid, y_valid, valid_missing = prepare_rank_matrix(valid, schema)
            eval_set = [(x_valid, y_valid)]
            eval_group = [group_valid]

    callbacks: list[Any] = []
    if eval_set is not None and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        min_child_samples=1,
        min_split_gain=0.0,
        num_leaves=31,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        importance_type="gain",
    )
    fit_kwargs: dict[str, Any] = {
        "X": x_train,
        "y": y_train.astype(float),
        "group": group_train,
    }
    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    if eval_set is not None:
        fit_kwargs["eval_X"] = eval_set[0][0]
        fit_kwargs["eval_y"] = eval_set[0][1]
        fit_kwargs["eval_group"] = eval_group
        fit_kwargs["eval_at"] = [12]
    model.fit(**fit_kwargs)
    best_iteration = int(getattr(model, "best_iteration_", 0) or n_estimators)
    schema = RankerSchema(
        feature_columns=schema.feature_columns,
        defaults=schema.defaults,
        label_column=schema.label_column,
        group_column=schema.group_column,
        best_iteration=best_iteration,
        n_estimators=n_estimators,
    )
    importance = {
        column: float(gain)
        for column, gain in zip(feature_columns, model.booster_.feature_importance(importance_type="gain"))
    }
    metrics = {
        "n_train_rows": int(len(train)),
        "n_train_groups": int(len(group_train)),
        "n_valid_rows": int(eval_set[0][0].shape[0]) if eval_set is not None else 0,
        "best_iteration": best_iteration,
        "n_estimators": n_estimators,
        "train_missing_rates": train_missing,
        "valid_missing_rates": valid_missing,
        "feature_importance_gain": importance,
        "schema_version": RANKER_SCHEMA_VERSION,
    }
    return model, schema, metrics


def save_ranker_artifact(
    model: Any,
    schema: RankerSchema,
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> RankerArtifact:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.txt"
    schema_path = output / "feature_schema.json"
    metrics_path = output / "metrics.json"
    booster = getattr(model, "booster_", model)
    booster.save_model(str(model_path))
    schema_path.write_text(json.dumps(schema.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return RankerArtifact(model_path=model_path, schema_path=schema_path, metrics_path=metrics_path, schema=schema, metrics=metrics)


def load_ranker_schema(schema_path: str | Path) -> RankerSchema:
    payload = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return RankerSchema.from_json(payload)


def load_ranker_booster(model_path: str | Path):
    lgb = require_lightgbm()
    return lgb.Booster(model_file=str(model_path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fashionrec ranker-train", description="Train LightGBM LambdaRank on ranking parquet tables")
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--valid-parquet", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--early-stopping", type=int, default=DEFAULT_EARLY_STOPPING)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if not args.train_parquet.is_file():
        raise FileNotFoundError(f"train parquet not found: {args.train_parquet}")
    train_frame = pd.read_parquet(args.train_parquet)
    valid_frame = None
    if args.valid_parquet is not None:
        if not args.valid_parquet.is_file():
            raise FileNotFoundError(f"valid parquet not found: {args.valid_parquet}")
        valid_frame = pd.read_parquet(args.valid_parquet)
    model, schema, metrics = train_lambdarank(
        train_frame,
        valid_frame,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    artifact = save_ranker_artifact(model, schema, metrics, args.output_dir)
    print(json.dumps({"model_path": str(artifact.model_path), "schema_path": str(artifact.schema_path), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

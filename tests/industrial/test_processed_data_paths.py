"""Tests for the baseline and run-scoped processed-data path contract."""

from pathlib import Path

from fashionrec.industrial.data.paths import DEFAULT_DATA_DIR, ProcessedDataPaths
from fashionrec.industrial.recall import channel_registry as registry


def test_default_processed_paths_keep_the_baseline_root() -> None:
    paths = ProcessedDataPaths.from_root()

    assert paths.root == DEFAULT_DATA_DIR == Path("data/processed")
    assert paths.train_inter == Path("data/processed/hm/hm.train.inter")
    assert paths.valid_inter == Path("data/processed/hm/hm.valid.inter")
    assert paths.test_inter == Path("data/processed/hm/hm.test.inter")
    assert paths.seq_train_inter == Path("data/processed/hm_seq/hm_seq.train.inter")
    assert paths.seq_item == Path("data/processed/hm_seq/hm_seq.item")


def test_custom_processed_paths_stay_under_one_run_root(tmp_path: Path) -> None:
    run_data = tmp_path / "outputs" / "runs" / "vnext-1" / "data"
    paths = ProcessedDataPaths.from_root(run_data)

    assert paths.root == run_data
    assert paths.model_train_inter == run_data / "hm" / "hm.model_train.inter"
    assert paths.valid_inter == run_data / "hm" / "hm.valid.inter"
    assert paths.seq_test_inter == run_data / "hm_seq" / "hm_seq.test.inter"


def test_rule_registry_uses_the_supplied_run_item_features(tmp_path: Path, monkeypatch) -> None:
    history = tmp_path / "hm.train.inter"
    history.write_text("user_id:token\titem_id:token\ttimestamp:float\n", encoding="utf-8")
    item_file = tmp_path / "hm_seq.item"
    item_file.write_text("item_id:token\n", encoding="utf-8")
    captured: dict[str, Path | None] = {}

    monkeypatch.setattr(registry, "build_popular_index", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(registry, "build_user_cohort_lookup", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(registry, "build_item2item_index", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(registry, "build_repurchase_index", lambda *_args, **_kwargs: object())

    def build_category(*_args, **kwargs):
        captured["item_file"] = kwargs.get("item_file")
        return object()

    monkeypatch.setattr(registry, "build_category_popular_index", build_category)
    registry.build_rule_channel_registry([history], item_file=item_file)

    assert captured["item_file"] == item_file

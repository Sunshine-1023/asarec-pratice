"""Select the SASRecF checkpoint on complete valid user-week MAP@K only."""

from __future__ import annotations

import argparse
from pathlib import Path

from fashionrec.baseline.data.paths import ProcessedDataPaths
from fashionrec.baseline.models.sasrecf.checkpoint_selection import score_recall_csv, select_checkpoint_by_score
from fashionrec.baseline.models.sasrecf.checkpoints import discover_checkpoint_candidates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fashionrec select-checkpoint", description="Select SASRecF checkpoint by valid user-week MAP@K")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed dataset root; defaults to data/processed.")
    parser.add_argument("--valid-inter", type=Path, default=None, help="Explicit valid split override.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline/models/sasrecf.yaml"))
    parser.add_argument("--recall-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--selected-model-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args(argv)
    data_paths = ProcessedDataPaths.from_root(args.data_dir)
    valid_inter = args.valid_inter or data_paths.valid_inter

    candidates = discover_checkpoint_candidates(args.checkpoint_dir)
    args.recall_dir.mkdir(parents=True, exist_ok=True)

    def score_candidate(checkpoint: Path) -> float:
        # Keep RecBole optional for import/help/tests; model loading happens only during real selection.
        from fashionrec.baseline.models.sasrecf.recall_service import export_sasrec_recall

        recall_path = args.recall_dir / f"{checkpoint.stem}_valid.csv"
        export_sasrec_recall(
            eval_split="valid",
            model_file=checkpoint,
            output_path=recall_path,
            top_k=args.top_k,
            config_path=args.config,
            channel="sasrecf",
            data_dir=data_paths.root,
        )
        return score_recall_csv(valid_inter, recall_path, k=args.top_k)

    selected = select_checkpoint_by_score(
        candidates,
        score_candidate,
        output_json=args.output_json,
        selected_model_path=args.selected_model_path,
        k=args.top_k,
    )
    print(f"Selected checkpoint: {selected}")
    print(f"Selection report: {args.output_json}")


if __name__ == "__main__":
    main()

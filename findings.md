# Findings & Decisions

## Requirements
- Refactor the overall project structure, logic, and workflow according to the architecture review.
- Ignore model training scores during this work.
- Preserve existing project behavior and user-owned uncommitted changes where possible.
- Keep current command-line entrypoints usable.
- Prepare a clean boundary for a later LightGBM LambdaRank ranking stage.

## Research Findings
- The top-level directory split (`data`, `recall`, `fusion`, `evaluate`, `experiment`) is reasonable, but the runtime remains script-driven and shares global output paths.
- `run_data_prep.py` imports sequence helpers from the training entrypoint `run_sasrec.py`, reversing the desired dependency direction.
- `src/evaluate/offline_eval.py` owns target loading, recall construction, candidate generation, fusion, metric computation, I/O, and CLI behavior.
- `configs/experiment.yaml` is currently consumed by data prep and baseline reporting, but is not propagated through training, recall export, weight search, or offline evaluation.
- `backtest_windows`, `union_top_k`, and the experiment seed are not fully consumed by the main pipeline.
- Activity-tier rules exist both in experiment config and `weighted_fusion.py`; the default test only verifies they currently match.
- Category Popular emits zero-padded item IDs, while Popular/Item2Item/history commonly emit unpadded IDs. Metrics canonicalize IDs, but fusion does not, so one physical item can occupy multiple candidate identities.
- Missing sequence recall and missing weights can silently degrade or fall back, which makes formal experiments ambiguous.
- Recall candidates are either exported by `run_rule_recall.py` or recomputed inside offline evaluation; there is no single materialized candidate contract.
- Output files are global (`outputs/recommendations`, `outputs/evaluation`, `outputs/checkpoints`) and can mix artifacts from different data/config versions.
- The current test suite passes 28 tests and provides a solid base for incremental refactoring.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Add `src/domain` | Central home for ID and candidate contracts used by all layers |
| Add `src/experiment/context.py` and `artifacts.py` | One run identity and explicit run-scoped paths |
| Add `src/data/build_sequences.py` | Remove data-prep dependency on the training entrypoint |
| Add `src/candidates` | Candidate union and schema are separate from recall and ranking |
| Add `src/ranking` | Weighted RRF and future LightGBM share one interface |
| Add recall registry/generator | All channels produce the same schema and can be configured consistently |
| Keep legacy wrappers during the transition | Avoid breaking README commands and existing use |
| Default formal orchestration to strict dependencies | Silent fallback remains only as an explicit compatibility/debug option |

## Final Architecture Outcomes
- All numeric H&M item IDs now use one ten-character representation at ingestion, recall, fusion, candidate, and metric boundaries.
- Sequence sample preparation is a pure data-layer service; neither data prep nor lower-level modules import a training entrypoint.
- Rule recall export, weight search, and evaluation share the same recall registry and candidate generator.
- Formal runs materialize four-channel candidates once and reuse that fixed artifact for search and evaluation.
- Weighted RRF implements a replaceable ranking interface; a LightGBM LambdaRank-ready feature table and group-size contract are available without training a model.
- `run_pipeline.py` resolves one experiment config and writes isolated checkpoints, recall, candidates, ranking outputs, evaluation files, and manifests under one run ID.
- Legacy `run_*.py` commands remain usable with their existing global defaults; formal pipeline mode is strict by default.

## Intentionally Deferred
- No SASRecF or LightGBM training was executed, per user scope.
- No score comparison or hyperparameter search was performed.
- A concrete `LGBMRanker` training adapter can be added next on top of `src/ranking/features.py`; the data and orchestration boundaries are now ready.
- Existing user-owned dirty-worktree changes and untracked documents were preserved.

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Worktree contains both tracked modifications and new files from prior work | Inspect before editing and avoid overwriting unrelated content |
| Existing metric canonicalization hides upstream ID inconsistencies | Move normalization to ingestion/candidate boundaries and test fusion de-duplication |

## Resources
- `configs/experiment.yaml`
- `run_pipeline.py`
- `run_data_prep.py`
- `run_sasrec.py`
- `src/evaluate/offline_eval.py`
- `src/fusion/weighted_fusion.py`
- `src/recall/category_popular.py`
- `.hermes/plans/2026-08-15_173300-fashionrec-optimization.md`

## Visual/Browser Findings
- No browser or image resources were used.

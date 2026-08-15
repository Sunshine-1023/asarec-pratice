# Progress Log

## Session: 2026-08-15

### Phase 1: Preserve Context and Define Contracts
- **Status:** complete
- **Started:** 2026-08-15
- Actions taken:
  - Reviewed the current project tree, import dependencies, configuration usage, pipeline order, recall implementations, fusion behavior, and tests.
  - Ran the existing test suite: 28 tests passed.
  - Identified cross-channel item-ID inconsistency, partial config propagation, shared global artifacts, silent fallbacks, and entrypoint-to-entrypoint coupling.
  - Created persistent planning files for the refactor.
  - Added canonical ID and candidate contracts under `src/domain`.
  - Added run-scoped artifact and immutable resolved-config context under `src/experiment`.
  - Added focused contract tests for domain objects and run context.
  - Propagated the canonical ten-character item ID through splitting, sequence preparation, Popular, Item2Item, ItemCF, fusion, targets, and metrics.
  - Added an integration test proving padded and unpadded representations fuse into one candidate.
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)
  - `src/domain/__init__.py` (created)
  - `src/domain/ids.py` (created)
  - `src/domain/candidates.py` (created)
  - `src/experiment/artifacts.py` (created)
  - `src/experiment/context.py` (created)
  - `tests/test_domain_contracts.py` (created)
  - `tests/test_run_context.py` (created)

### Phase 2: Decouple Data Preparation and Training
- **Status:** complete
- Actions taken:
  - Moved causal train/valid/test sequence conversion into `src/data/build_sequences.py`.
  - Changed `run_data_prep.py` to depend directly on the data layer instead of importing `run_sasrec.py`.
  - Kept legacy sequence helper names exported by `run_sasrec.py` as compatibility aliases.
  - Added tests for canonical IDs, train rolling history, valid non-rolling history, and test consumption of valid history.
- Files created/modified:
  - `src/data/build_sequences.py` (created)
  - `run_data_prep.py`
  - `run_sasrec.py`
  - `tests/test_build_sequences.py` (created)

### Phase 3: Unify Recall and Candidate Generation
- **Status:** complete
- Actions taken:
  - Added a recall channel protocol, rule-channel adapters, registry, and precomputed-channel adapter.
  - Added one candidate generator that emits the immutable shared `Candidate` schema.
  - Added deterministic candidate union/de-duplication while preserving multi-channel evidence for ranking features.
  - Changed batch rule export and offline evaluation to share the same registry and generator.
  - Added strict sequence-recall dependency support to formal evaluation while preserving legacy fallback by default.
- Files created/modified:
  - `src/recall/base.py` (created)
  - `src/recall/registry.py` (created)
  - `src/recall/generator.py` (created)
  - `src/recall/rule_recall_export.py`
  - `src/candidates/__init__.py` (created)
  - `src/candidates/union.py` (created)
  - `src/evaluate/offline_eval.py`
  - `tests/test_candidate_pipeline.py` (created)

### Phase 4: Separate Ranking, Evaluation, and Orchestration
- **Status:** complete
- Actions taken:
  - Added a generic ranking protocol and moved weighted RRF behind `WeightedRRFRanker`.
  - Added a deterministic one-row-per-user-item feature table and group-size helper for LightGBM LambdaRank.
  - Added optional strict consumption of materialized four-channel candidates in weight search and evaluation.
  - Added a pure config-driven pipeline planner that propagates candidate limits, final K, seed, and artifact paths.
  - Made formal pipeline outputs run-scoped, including checkpoints, recall, candidates, ranking results, metrics, and manifests.
  - Preserved existing `run_*.py` commands and compatibility mode; lazy-loaded RecBole so non-training CLI inspection does not require it.
- Files created/modified:
  - `src/ranking/__init__.py` (created)
  - `src/ranking/base.py` (created)
  - `src/ranking/weighted_rrf.py` (created)
  - `src/ranking/features.py` (created)
  - `src/pipeline/__init__.py` (created)
  - `src/pipeline/orchestrator.py` (created)
  - `src/evaluate/offline_eval.py`
  - `src/evaluate/weight_search.py`
  - `src/experiment/context.py`
  - `run_pipeline.py`
  - `run_sasrec.py`
  - `tests/test_ranking.py` (created)
  - `tests/test_pipeline_orchestrator.py` (created)

### Phase 5: Verification and Documentation
- **Status:** complete
- Actions taken:
  - Added `docs/ARCHITECTURE.md` and updated the README structure, flow, run-scoped outputs, and ranking boundary.
  - Ran `git diff --check` successfully after fixing one pre-existing README trailing-space line.
  - Parsed all 51 project Python files with `ast.parse` without syntax errors.
  - Verified no source/data module imports a top-level training entrypoint.
  - Ran help smoke tests for data prep, SASRec, rule recall, weight search, offline evaluation, and full pipeline without training.
  - Re-ran the complete test suite: 44 tests passed.
- Files created/modified:
  - `docs/ARCHITECTURE.md` (created)
  - `README.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Existing project suite | `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q` | All existing tests pass | 28 passed | ✓ |
| Phase 1 full suite | same command | Contract propagation does not regress behavior | 37 passed | ✓ |
| Phase 2 full suite | same command | Data/training decoupling remains compatible | 39 passed | ✓ |
| Phase 3 full suite | same command | Shared recall/candidate path remains compatible | 40 passed | ✓ |
| Phase 4 full suite | same command | Ranking/orchestration refactor remains compatible | 44 passed | ✓ |
| Final full suite | same command | All refactor phases remain green | 44 passed | ✓ |
| Python syntax audit | AST parse all `src/**/*.py` and `run_*.py` | No syntax errors | 51 files parsed | ✓ |
| CLI smoke | `--help` on six main entrypoints | No training/import failure | All passed | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-08-15 | Bulk ID-contract patch failed context verification before changing files | 1 | No partial edits were applied; switch to smaller patches after inspecting exact headers |
| 2026-08-15 | Focused suite: time-split test expected `1`, `2` after IDs became canonical | 1 | Update test expectation to `0000000001`, `0000000002` |
| 2026-08-15 | CLI smoke for `run_sasrec.py --help` failed with `ModuleNotFoundError: recbole` | 1 | Lazy-load RecBole only when training actually starts; no environment installation required |
| 2026-08-15 | Combined documentation patch failed on one README context line | 1 | Split documentation creation and README edits into targeted patches |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Refactor complete |
| Where am I going? | Optional next phase: implement and validate LightGBM LambdaRank training |
| What's the goal? | Contract-driven, run-scoped, backward-compatible recommendation pipeline |
| What have I learned? | See `findings.md` |
| What have I done? | Completed contracts, data decoupling, unified candidates, ranking boundary, run-scoped orchestration, tests, and docs |

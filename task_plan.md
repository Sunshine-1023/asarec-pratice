# Task Plan: FashionRec-Transformer Architecture Refactor

## Goal
Refactor the project into a contract-driven, run-scoped recommendation pipeline with consistent IDs, explicit artifacts, decoupled data/recall/ranking/evaluation stages, and backward-compatible CLI entrypoints.

## Next Step
Refactor complete; hand off the new structure, verification evidence, and intentionally deferred model-training work.

## Current Phase
Complete

## Phases

### Phase 1: Preserve Context and Define Contracts
- [x] Review the current tree, imports, workflow, and tests
- [x] Record structural risks and compatibility constraints
- [x] Add canonical ID, candidate schema, artifact paths, and run context
- [x] Add unit tests for shared contracts
- [x] Propagate canonical IDs through existing pipeline boundaries
- [x] Add a fusion de-duplication integration test
- **Status:** complete

### Phase 2: Decouple Data Preparation and Training
- [x] Move sequence-sample preparation out of `run_sasrec.py`
- [x] Stop data preparation importing a training entrypoint or private helpers
- [x] Preserve existing CLI behavior
- [x] Add sequence preparation tests
- **Status:** complete

### Phase 3: Unify Recall and Candidate Generation
- [x] Add a recall channel interface and registry
- [x] Normalize all channel outputs to one candidate schema
- [x] Add candidate union/de-duplication
- [x] Make rule export and evaluation share the same candidate generator
- [x] Add ID and candidate integration tests
- **Status:** complete

### Phase 4: Separate Ranking, Evaluation, and Orchestration
- [x] Extract weighted RRF ranker from evaluation flow
- [x] Split prediction evaluation/reporting from candidate generation
- [x] Add run-scoped artifact directories and strict dependency checks
- [x] Propagate the experiment config through pipeline stages
- [x] Keep legacy `run_*.py` commands compatible
- **Status:** complete

### Phase 5: Verification and Documentation
- [x] Run the complete test suite
- [x] Run CLI/import smoke tests without model training
- [x] Review the final dependency direction and dirty-worktree overlap
- [x] Update architecture/process documentation
- [x] Summarize changed files and remaining follow-up work
- **Status:** complete

## Key Questions
1. How can the architecture be improved without overwriting the user's existing uncommitted baseline work?
2. Which compatibility wrappers must remain so documented commands continue to work?
3. How should artifacts be identified so stale checkpoints, recall files, and weights cannot mix silently?
4. What is the smallest useful ranking boundary that supports both weighted RRF and later LightGBM LambdaRank?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use a gradual refactor with compatibility wrappers | The worktree already contains substantial user-owned changes and documented commands |
| Keep internal item IDs canonical and format submission IDs only at the boundary | Prevent cross-channel duplicate candidates caused by leading-zero differences |
| Introduce run-scoped artifacts but retain legacy output defaults as opt-in compatibility | Enables reproducibility without abruptly breaking existing scripts |
| Materialize a common candidate table before ranking/evaluation | Weighted fusion and future LightGBM must consume identical candidates |
| Keep model training details in model YAML referenced by experiment config | Avoid duplicating RecBole hyperparameters in the orchestration config |
| Do not perform Git operations | The user asked for a refactor, not Git changes |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Bulk ID-contract patch did not match `build_item_features.py` import context | 1 | Inspect exact file headers and apply smaller targeted patches |
| Time-split sorting test expected legacy unpadded IDs after canonicalization | 1 | Update the assertion to the new ten-character shared ID contract |
| `run_sasrec.py --help` imported RecBole eagerly and failed in the current environment | 1 | Move optional RecBole imports into the actual training function so CLI/help and data modules remain inspectable |
| Combined architecture-doc/README patch missed an exact README sentence | 1 | No partial README edit applied; add the document and README sections in smaller patches |

## Notes
- Existing user modifications must be preserved.
- No model training or score comparison is required for this refactor.
- Current baseline before refactor: 28 tests pass.

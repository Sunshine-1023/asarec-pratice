from __future__ import annotations

from fashionrec.baseline.evaluation.offline_eval import main as run_evaluation
from fashionrec.shared.runtime.argv import normalized_argv, reject_options


def main(argv: list[str] | None = None) -> None:
    args = normalized_argv(argv)
    reject_options(
        args,
        ("--labels-dir", "--ranker-scored-csv"),
        application="baseline evaluation",
    )
    run_evaluation(args)

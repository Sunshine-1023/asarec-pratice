from __future__ import annotations

from fashionrec.industrial.evaluation.offline_eval import main as run_evaluation
from fashionrec.shared.runtime.argv import normalized_argv, require_option


def main(argv: list[str] | None = None) -> None:
    args = normalized_argv(argv)
    require_option(args, "--labels-dir", application="industrial evaluation")
    require_option(args, "--ranker-scored-csv", application="industrial evaluation")
    run_evaluation(args)

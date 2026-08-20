from __future__ import annotations

from fashionrec.baseline.evaluation.weight_search import main as run_weight_search
from fashionrec.shared.runtime.argv import normalized_argv, reject_options


def main(argv: list[str] | None = None) -> None:
    args = normalized_argv(argv)
    reject_options(args, ("--labels-dir",), application="baseline weight search")
    run_weight_search(args)

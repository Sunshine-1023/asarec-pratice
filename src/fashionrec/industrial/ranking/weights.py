from __future__ import annotations

from fashionrec.industrial.evaluation.weight_search import main as run_weight_search
from fashionrec.shared.runtime.argv import normalized_argv, require_option


def main(argv: list[str] | None = None) -> None:
    args = normalized_argv(argv)
    require_option(args, "--labels-dir", application="industrial weight search")
    run_weight_search(args)

from __future__ import annotations

from fashionrec.industrial.recall.registry import INDUSTRIAL_RULE_CHANNELS
from fashionrec.industrial.recall.service import main as run_candidate_service
from fashionrec.shared.runtime.argv import force_option, normalized_argv


def main(argv: list[str] | None = None) -> None:
    channels = ",".join(INDUSTRIAL_RULE_CHANNELS)
    run_candidate_service(force_option(normalized_argv(argv), "--channels", channels))

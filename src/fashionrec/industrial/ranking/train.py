from __future__ import annotations

from fashionrec.industrial.models.lambdarank.train import *  # noqa: F403
from fashionrec.industrial.models.lambdarank.train import main as run_ranker_training


def main(argv: list[str] | None = None) -> None:
    run_ranker_training(argv)

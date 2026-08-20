from __future__ import annotations

from fashionrec.industrial.models.lambdarank.predict import *  # noqa: F403
from fashionrec.industrial.models.lambdarank.predict import main as run_ranker_prediction


def main(argv: list[str] | None = None) -> None:
    run_ranker_prediction(argv)

"""Public command surface for the isolated baseline application."""

from __future__ import annotations

from fashionrec.shared.runtime.dispatch import ApplicationCommand, dispatch


COMMANDS = {
    "pipeline": ApplicationCommand("fashionrec.baseline.pipeline.command", "Run the complete baseline DAG"),
    "data": ApplicationCommand("fashionrec.baseline.data.command", "Prepare baseline interaction data"),
    "train": ApplicationCommand("fashionrec.baseline.models.sasrecf.train", "Train baseline SASRecF"),
    "select-checkpoint": ApplicationCommand(
        "fashionrec.baseline.models.sasrecf.select_checkpoint", "Select baseline SASRecF checkpoint"
    ),
    "recall": ApplicationCommand("fashionrec.baseline.models.sasrecf.recall", "Export baseline SASRecF recall"),
    "candidates": ApplicationCommand("fashionrec.baseline.recall.command", "Materialize fixed four-channel candidates"),
    "weights": ApplicationCommand("fashionrec.baseline.ranking.weights", "Search baseline RRF weights"),
    "evaluate": ApplicationCommand("fashionrec.baseline.evaluation.command", "Evaluate baseline line-level labels"),
}


def main(argv: list[str] | None = None) -> int:
    return dispatch(application="baseline", commands=COMMANDS, argv=argv)

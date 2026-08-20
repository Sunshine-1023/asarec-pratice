"""Public command surface for the isolated industrial application."""

from __future__ import annotations

from fashionrec.shared.runtime.dispatch import ApplicationCommand, dispatch


COMMANDS = {
    "pipeline": ApplicationCommand("fashionrec.industrial.pipeline.command", "Run the complete industrial DAG"),
    "data": ApplicationCommand("fashionrec.industrial.data.command", "Build events, baskets, labels and PIT data"),
    "train": ApplicationCommand("fashionrec.industrial.models.sasrecf.train", "Train industrial SASRecF channel"),
    "select-checkpoint": ApplicationCommand(
        "fashionrec.industrial.models.sasrecf.select_checkpoint", "Select industrial SASRecF checkpoint"
    ),
    "recall": ApplicationCommand("fashionrec.industrial.models.sasrecf.recall", "Export industrial sequence recall"),
    "candidates": ApplicationCommand("fashionrec.industrial.recall.command", "Materialize expanded multi-recall candidates"),
    "weights": ApplicationCommand("fashionrec.industrial.ranking.weights", "Search next-basket RRF control weights"),
    "ranker-dataset": ApplicationCommand(
        "fashionrec.industrial.ranking.dataset_materialization", "Build causal LambdaRank tables"
    ),
    "ranker-train": ApplicationCommand("fashionrec.industrial.ranking.train", "Train LightGBM LambdaRank"),
    "ranker-predict": ApplicationCommand("fashionrec.industrial.ranking.predict", "Score candidates with LambdaRank"),
    "evaluate": ApplicationCommand("fashionrec.industrial.evaluation.command", "Evaluate next-basket RRF vs LambdaRank"),
}


def main(argv: list[str] | None = None) -> int:
    return dispatch(application="industrial", commands=COMMANDS, argv=argv)

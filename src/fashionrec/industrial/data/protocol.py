"""Validation for the industrial next-basket/LambdaRank protocol."""

from fashionrec.experiment.context import RunContext


def validate_context(context: RunContext) -> None:
    cfg = context.config
    if context.artifacts.profile != "industrial":
        raise ValueError("industrial application requires an industrial run context")
    if cfg.ranking.library != "lightgbm" or cfg.ranking.objective != "lambdarank":
        raise ValueError("industrial application currently supports only lightgbm + lambdarank")
    if cfg.label.target_mode != "next_basket":
        raise ValueError("industrial application requires label.target_mode=next_basket")

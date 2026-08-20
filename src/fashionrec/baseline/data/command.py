"""Baseline-owned data command with industrial artifacts explicitly forbidden."""

from __future__ import annotations

from fashionrec.baseline.settings import EXPERIMENT_CONFIG, SASRECF_CONFIG
from fashionrec.baseline.data.service import main as run_shared_data_service
from fashionrec.shared.runtime.argv import force_option, normalized_argv, reject_options


INDUSTRIAL_OPTIONS = (
    "--build-events",
    "--build-baskets",
    "--build-labels",
    "--build-user-features",
    "--build-cross-features",
    "--candidates",
)


def main(argv: list[str] | None = None) -> None:
    args = normalized_argv(argv)
    reject_options(args, INDUSTRIAL_OPTIONS, application="baseline data")
    args = force_option(args, "--config", str(SASRECF_CONFIG))
    args = force_option(args, "--experiment-config", str(EXPERIMENT_CONFIG))
    run_shared_data_service(args)

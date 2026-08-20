"""Industrial-owned data command with mandatory basket/PIT artifacts."""

from __future__ import annotations

from fashionrec.industrial.data.service import main as run_shared_data_service
from fashionrec.industrial.settings import EXPERIMENT_CONFIG, SASRECF_CONFIG
from fashionrec.shared.runtime.argv import ensure_flag, force_option, normalized_argv


REQUIRED_FLAGS = (
    "--build-events",
    "--build-baskets",
    "--build-labels",
    "--build-user-features",
)


def main(argv: list[str] | None = None) -> None:
    args = force_option(normalized_argv(argv), "--config", str(SASRECF_CONFIG))
    args = force_option(args, "--experiment-config", str(EXPERIMENT_CONFIG))
    for flag in REQUIRED_FLAGS:
        args = ensure_flag(args, flag)
    run_shared_data_service(args)

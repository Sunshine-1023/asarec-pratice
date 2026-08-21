"""Explicit top-level router for the two FashionRec applications."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class CommandSpec:
    module: str
    description: str


COMMANDS: dict[str, CommandSpec] = {
    "baseline": CommandSpec("fashionrec.baseline.cli", "Run a Baseline application command"),
    "industrial": CommandSpec("fashionrec.industrial.cli", "Run an Industrial application command"),
    "profile-data": CommandSpec(
        "fashionrec.industrial.data.profile",
        "Profile raw transactions, customers and articles",
    ),
}


def _command_help() -> str:
    width = max(len(name) for name in COMMANDS)
    return "\n".join(
        f"  {name:<{width}}  {spec.description}" for name, spec in COMMANDS.items()
    )


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="fashionrec",
        description="FashionRec application selector",
        epilog=f"commands:\n{_command_help()}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", help="Application or utility command")
    if not raw_args or raw_args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command, command_args = raw_args[0], raw_args[1:]
    spec = COMMANDS.get(command)
    if spec is None:
        parser.error(
            f"unknown command: {command}; choose 'baseline', 'industrial', or 'profile-data'"
        )
    module = importlib.import_module(spec.module)
    command_main: Callable[[list[str] | None], object] | None = getattr(module, "main", None)
    if not callable(command_main):
        raise RuntimeError(f"Command module has no callable main(): {spec.module}")
    command_main(command_args)
    return 0

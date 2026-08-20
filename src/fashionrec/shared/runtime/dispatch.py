"""Lazy command dispatcher reused by the two isolated applications."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class ApplicationCommand:
    module: str
    description: str


def dispatch(
    *,
    application: str,
    commands: dict[str, ApplicationCommand],
    argv: list[str] | None = None,
) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog=f"fashionrec.{application}",
        description=f"FashionRec {application} application",
        epilog="commands:\n"
        + "\n".join(f"  {name:<20} {spec.description}" for name, spec in commands.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", help="Application command")
    if not raw_args or raw_args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command, command_args = raw_args[0], raw_args[1:]
    spec = commands.get(command)
    if spec is None:
        parser.error(f"unknown {application} command: {command}")
    module = importlib.import_module(spec.module)
    command_main: Callable[[list[str] | None], object] | None = getattr(module, "main", None)
    if not callable(command_main):
        raise RuntimeError(f"Command module has no callable main(): {spec.module}")
    command_main(command_args)
    return 0

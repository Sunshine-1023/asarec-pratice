"""Small helpers for application-owned command wrappers."""

from __future__ import annotations

from collections.abc import Iterable


def normalized_argv(argv: list[str] | None) -> list[str]:
    return list(argv or [])


def has_option(argv: Iterable[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def reject_options(argv: Iterable[str], options: Iterable[str], *, application: str) -> None:
    forbidden = [option for option in options if has_option(argv, option)]
    if forbidden:
        raise ValueError(f"{application} does not allow options: {', '.join(forbidden)}")


def force_option(argv: list[str], option: str, value: str) -> list[str]:
    """Remove every existing value for one argparse option and append the owned value."""

    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            index += 2
            continue
        if token.startswith(f"{option}="):
            index += 1
            continue
        result.append(token)
        index += 1
    result.extend((option, value))
    return result


def ensure_flag(argv: list[str], flag: str) -> list[str]:
    return argv if has_option(argv, flag) else [*argv, flag]


def require_option(argv: Iterable[str], option: str, *, application: str) -> None:
    if "--help" in argv or "-h" in argv:
        return
    if not has_option(argv, option):
        raise ValueError(f"{application} requires {option}")

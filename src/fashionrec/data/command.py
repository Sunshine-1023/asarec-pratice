"""Compatibility facade for the former shared data command."""

from fashionrec.industrial.data.service import *  # noqa: F403
from fashionrec.industrial.data.service import main

__all__ = ["main"]

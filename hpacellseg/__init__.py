"""Provide a package for NEW_REPO."""
from pathlib import Path

__version__ = (Path(__file__).parent / "VERSION").read_text().strip()

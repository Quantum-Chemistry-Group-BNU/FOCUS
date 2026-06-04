"""Expose the bundled CAMPS code as the historical top-level `camps` package."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "pyfocus" / "camps")]

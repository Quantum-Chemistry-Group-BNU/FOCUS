import sys

from loguru import logger

__all__ = ["dist_print"]

def _dist_print(values: object, master: bool = False) -> None:
    s = values
    sys.stdout.write(s)

def dist_print(message) -> None:
    flags = message.record["extra"].get("master", False)
    _dist_print(message, flags)

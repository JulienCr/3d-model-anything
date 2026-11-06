"""French number formatting shared by the plan, the model labels and the report."""
from __future__ import annotations


def fr(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")

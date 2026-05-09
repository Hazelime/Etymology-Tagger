from __future__ import annotations

from pathlib import Path


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def assert_under_budget(root: Path, budget_gb: float, incoming_bytes: int = 0) -> None:
    budget_bytes = int(budget_gb * 1024**3)
    projected = directory_size(root) + incoming_bytes
    if projected > budget_bytes:
        raise RuntimeError(
            f"Projected project storage is {projected / 1024**3:.2f} GB, "
            f"above the configured {budget_gb:.2f} GB budget. "
            "Ask for confirmation before increasing the budget."
        )

from __future__ import annotations

import json
from pathlib import Path

# Global constants for project-wide path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(path: str | Path) -> dict:
    """Loads the project configuration JSON and injects the project root path."""
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_project_root"] = str(PROJECT_ROOT)
    return config

def resolve(root: str | Path, rel_path: str | Path) -> Path:
    """Safely resolves a relative path against the project root."""
    return (Path(root) / rel_path).resolve()

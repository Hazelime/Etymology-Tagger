from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import PROJECT_ROOT

SPACE_FILES = [
    "app.py",
    "requirements.txt",
    "pyproject.toml",
    "configs/prototype.json",
    "data/processed/etymology_records.jsonl",
    "data/processed/labels.json",
    "models/fasttext_subset.vec",
    "models/etymology_tagger.pt",
    "models/metadata.json",
]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist/space")
    args = parser.parse_args()
    out_dir = (PROJECT_ROOT / args.out).resolve()
    if not str(out_dir).startswith(str(PROJECT_ROOT.resolve())):
        raise RuntimeError(f"Refusing to write outside project root: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for rel_path in SPACE_FILES:
        copy_file(PROJECT_ROOT / rel_path, out_dir / rel_path)
    shutil.copytree(PROJECT_ROOT / "src", out_dir / "src")
    copy_file(PROJECT_ROOT / "space_README.md", out_dir / "README.md")
    print(f"Wrote Space package to {out_dir}")


if __name__ == "__main__":
    main()

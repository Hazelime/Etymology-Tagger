from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import PROJECT_ROOT


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist/dataset")
    args = parser.parse_args()
    out_dir = (PROJECT_ROOT / args.out).resolve()
    if not str(out_dir).startswith(str(PROJECT_ROOT.resolve())):
        raise RuntimeError(f"Refusing to write outside project root: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    copy_file(PROJECT_ROOT / "data/processed/etymology_records.jsonl", out_dir / "etymology_records.jsonl")
    copy_file(PROJECT_ROOT / "data/processed/labels.json", out_dir / "labels.json")
    copy_file(PROJECT_ROOT / "data/processed/extraction_stats.json", out_dir / "extraction_stats.json")
    copy_file(PROJECT_ROOT / "dataset_card.md", out_dir / "README.md")
    print(f"Wrote dataset package to {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import PROJECT_ROOT

def copy_file(src: Path, dst: Path) -> None:
    """Helper to copy a file and ensure the destination directory exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main() -> None:
    """
    Prepares a clean distribution package for Hugging Face Datasets.
    
    This script:
    1. Clears the 'dist/dataset' output directory.
    2. Copies the processed JSONL records and label metadata.
    3. Replaces the generic 'dataset_card.md' with the 'README.md' required 
       by Hugging Face.
       
    The resulting folder can be uploaded directly to a Hugging Face Dataset repository.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist/dataset", help="Output directory for the dataset package")
    args = parser.parse_args()
    
    out_dir = (PROJECT_ROOT / args.out).resolve()
    
    # Security check to prevent writing outside the project workspace
    if not str(out_dir).startswith(str(PROJECT_ROOT.resolve())):
        raise RuntimeError(f"Refusing to write outside project root: {out_dir}")
        
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    # Copy core dataset files
    copy_file(PROJECT_ROOT / "data/processed/etymology_records.jsonl", out_dir / "etymology_records.jsonl")
    copy_file(PROJECT_ROOT / "data/processed/labels.json", out_dir / "labels.json")
    copy_file(PROJECT_ROOT / "data/processed/extraction_stats.json", out_dir / "extraction_stats.json")
    
    # Map the dataset card to the root README for Hugging Face display
    copy_file(PROJECT_ROOT / "dataset_card.md", out_dir / "README.md")
    
    print(f"Success: Wrote dataset package to {out_dir}")

if __name__ == "__main__":
    main()

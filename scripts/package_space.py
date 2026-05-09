from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import PROJECT_ROOT

# List of files required for the interactive Gradio app to run in Hugging Face Spaces.
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
    """Helper to copy a file and ensure the destination directory exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main() -> None:
    """
    Prepares a clean distribution package for Hugging Face Spaces.
    
    This script:
    1. Clears the 'dist/space' output directory.
    2. Copies the core app files, training metadata, and model weights.
    3. Bundles the local 'src' package.
    4. Replaces the generic 'space_README.md' with the 'README.md' required 
       by Hugging Face.
       
    The resulting folder can be uploaded directly to a Hugging Face Space repository.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dist/space", help="Output directory for the Space package")
    args = parser.parse_args()
    
    out_dir = (PROJECT_ROOT / args.out).resolve()
    
    # Security check to prevent writing outside the project workspace
    if not str(out_dir).startswith(str(PROJECT_ROOT.resolve())):
        raise RuntimeError(f"Refusing to write outside project root: {out_dir}")
        
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    # Copy essential files maintaining the directory structure
    for rel_path in SPACE_FILES:
        src_path = PROJECT_ROOT / rel_path
        if src_path.exists():
            copy_file(src_path, out_dir / rel_path)
        else:
            print(f"Warning: Skipping missing file {rel_path}")
            
    # Copy the entire source package
    shutil.copytree(PROJECT_ROOT / "src", out_dir / "src")
    
    # Map the space card to the root README for Hugging Face display
    copy_file(PROJECT_ROOT / "space_README.md", out_dir / "README.md")
    
    print(f"Success: Wrote Space package to {out_dir}")

if __name__ == "__main__":
    main()

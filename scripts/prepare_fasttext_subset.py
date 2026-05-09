from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, resolve
from etymology_tagger.extract import read_jsonl
from etymology_tagger.storage import assert_under_budget
from etymology_tagger.vectors import download_file, extract_vector_subset_from_zip

def main() -> None:
    """
    Downloads the official fastText pretrained vectors and extracts a compact subset.
    
    This is a critical preprocessing step that:
    1. Collects a vocabulary of all words found in our processed etymology records.
    2. Downloads the ~4GB 'wiki-news-300d-1M' zip archive from fasttext.cc.
    3. Streams the zip content to extract only the vectors relevant to our vocabulary.
    4. Deletes the large zip file to save disk space.
    
    This results in a ~80MB .vec file that is small enough for deployment but 
    retains the full semantic power of fastText for our specific domain.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    parser.add_argument("--keep-download", action="store_true", help="Do not delete the 4GB zip after extraction")
    args = parser.parse_args()
    config = load_config(args.config)
    
    root = Path(config["_project_root"])
    
    # 1. Build vocabulary from our records
    records = read_jsonl(resolve(config["_project_root"], config["records_path"]))
    vocab = set()
    for record in records:
        word = record["word"].lower().strip()
        vocab.add(word)
        # Also include constituent words to support averaging fallback for phrases
        for part in word.replace("-", " ").split():
            if part:
                vocab.add(part)
                
    # 2. Download and Extract
    download_path = resolve(config["_project_root"], config["model_dir"]) / "wiki-news-300d-1M.vec.zip"
    print(f"Downloading fastText vectors from {config['fasttext_url']}...")
    download_file(config["fasttext_url"], download_path, root, config["storage_budget_gb"])
    
    print(f"Subsetting vectors for {len(vocab)} unique terms...")
    found = extract_vector_subset_from_zip(
        download_path,
        resolve(config["_project_root"], config["vector_subset_path"]),
        vocab,
    )
    
    # 3. Cleanup
    if not args.keep_download:
        print("Cleaning up temporary zip file...")
        download_path.unlink(missing_ok=True)
        
    print(f"Success: Wrote vectors for {found}/{len(vocab)} dataset words.")

if __name__ == "__main__":
    main()

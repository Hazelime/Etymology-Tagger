from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, resolve
from etymology_tagger.extract import read_jsonl
from etymology_tagger.storage import assert_under_budget
from etymology_tagger.vectors import download_file, extract_vector_subset_from_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    parser.add_argument("--keep-download", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["_project_root"])
    assert_under_budget(root, config["storage_budget_gb"])
    records = read_jsonl(resolve(config["_project_root"], config["records_path"]))
    vocab = set()
    for record in records:
        word = record["word"].lower().strip()
        vocab.add(word)
        # Also include constituent words for averaging
        for part in word.replace("-", " ").split():
            if part:
                vocab.add(part)
    download_path = resolve(config["_project_root"], config["model_dir"]) / "wiki-news-300d-1M.vec.zip"
    download_file(config["fasttext_url"], download_path, root, config["storage_budget_gb"])
    found = extract_vector_subset_from_zip(
        download_path,
        resolve(config["_project_root"], config["vector_subset_path"]),
        vocab,
    )
    if not args.keep_download:
        download_path.unlink(missing_ok=True)
    assert_under_budget(root, config["storage_budget_gb"])
    print(f"Wrote vectors for {found}/{len(vocab)} dataset words.")


if __name__ == "__main__":
    main()

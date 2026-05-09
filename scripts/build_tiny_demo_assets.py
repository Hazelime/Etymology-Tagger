from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config, resolve
from etymology_tagger.extract import (
    remap_to_top_languages,
    select_labels,
)

# A small set of representative etymological records for testing.
SAMPLE_RECORDS = [
    {
        "word": "alcohol",
        "etymology_texts": ["Borrowed from French alcohol, derived ultimately from Arabic al-kuhl."],
        "pairs": [
            {"mechanism": "borrowed", "source_language": "French", "source_code": "fr"},
            {"mechanism": "derived", "source_language": "Arabic", "source_code": "ar"},
        ],
        "source_languages": ["French", "Arabic"],
        "mechanisms": ["borrowed", "derived"],
    },
    {
        "word": "sushi",
        "etymology_texts": ["Borrowed from Japanese sushi."],
        "pairs": [{"mechanism": "borrowed", "source_language": "Japanese", "source_code": "ja"}],
        "source_languages": ["Japanese"],
        "mechanisms": ["borrowed"],
    },
    {
        "word": "father",
        "etymology_texts": ["Inherited from Middle English fader, from Old English faeder."],
        "pairs": [
            {"mechanism": "inherited", "source_language": "English", "source_code": "enm"},
            {"mechanism": "inherited", "source_language": "English", "source_code": "ang"},
        ],
        "source_languages": ["English"],
        "mechanisms": ["inherited"],
    },
]

def main() -> None:
    """
    Smoke test script to generate a minimal dataset and fake vectors.
    
    This is used to verify that the project plumbing (data loading, model 
    instantiation, UI rendering) works correctly without requiring a 
    4GB fastText download.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Process the small sample set
    labels = select_labels(SAMPLE_RECORDS, config.get("top_n_languages", 0.01))
    records = [remap_to_top_languages(record, labels["source_languages"]) for record in SAMPLE_RECORDS]
    
    # Write tiny data files
    records_path = resolve(config["_project_root"], config["records_path"])
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    labels_path = resolve(config["_project_root"], config["labels_path"])
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    
    # Generate zero-vectors for the sample vocabulary
    vec_path = resolve(config["_project_root"], config["vector_subset_path"])
    with vec_path.open("w", encoding="utf-8") as f:
        f.write(f"{len(records)} 300\n")
        for r in records:
            zeros = " ".join(["0.0"] * 300)
            f.write(f"{r['word']} {zeros}\n")
            
    print(f"Success: Wrote tiny smoke-test assets to {records_path.parent}")

if __name__ == "__main__":
    main()

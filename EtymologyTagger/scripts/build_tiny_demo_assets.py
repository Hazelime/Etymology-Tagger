from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config, resolve
from etymology_tagger.extract import (
    remap_to_top_languages,
    select_labels,
    write_jsonl,
    write_labels,
)
from etymology_tagger.vectors import write_tiny_vectors

SAMPLE_RECORDS = [
    {
        "word": "alcohol",
        "display_word": "alcohol",
        "parts_of_speech": ["noun"],
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
        "display_word": "sushi",
        "parts_of_speech": ["noun"],
        "etymology_texts": ["Borrowed from Japanese sushi."],
        "pairs": [{"mechanism": "borrowed", "source_language": "Japanese", "source_code": "ja"}],
        "source_languages": ["Japanese"],
        "mechanisms": ["borrowed"],
    },
    {
        "word": "father",
        "display_word": "father",
        "parts_of_speech": ["noun"],
        "etymology_texts": ["Inherited from Middle English fader, from Old English faeder."],
        "pairs": [
            {"mechanism": "inherited", "source_language": "Middle English", "source_code": "enm"},
            {"mechanism": "inherited", "source_language": "Old English", "source_code": "ang"},
        ],
        "source_languages": ["Middle English", "Old English"],
        "mechanisms": ["inherited"],
    },
    {
        "word": "parabola",
        "display_word": "parabola",
        "parts_of_speech": ["noun"],
        "etymology_texts": ["Derived from Ancient Greek parabole."],
        "pairs": [{"mechanism": "derived", "source_language": "Ancient Greek", "source_code": "grc"}],
        "source_languages": ["Ancient Greek"],
        "mechanisms": ["derived"],
    },
    {
        "word": "skyscraper",
        "display_word": "skyscraper",
        "parts_of_speech": ["noun"],
        "etymology_texts": ["Calqued from French gratte-ciel."],
        "pairs": [{"mechanism": "calqued", "source_language": "French", "source_code": "fr"}],
        "source_languages": ["French"],
        "mechanisms": ["calqued"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    args = parser.parse_args()
    config = load_config(args.config)
    labels = select_labels(SAMPLE_RECORDS, config["top_n_languages"])
    records = [remap_to_top_languages(record, labels["source_languages"]) for record in SAMPLE_RECORDS]
    write_jsonl(resolve(config["_project_root"], config["records_path"]), records)
    write_labels(resolve(config["_project_root"], config["labels_path"]), labels)
    write_tiny_vectors(
        resolve(config["_project_root"], config["vector_subset_path"]),
        [record["word"] for record in records],
    )
    print(f"Wrote tiny dataset and vectors for {len(records)} words.")


if __name__ == "__main__":
    main()

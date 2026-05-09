from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config, resolve
from etymology_tagger.extract import (
    ExtractionStats,
    compact_record,
    load_jsonl_from_url,
    merge_records,
    remap_to_top_languages,
    select_labels,
)

def main() -> None:
    """
    ETL script to build the etymology dataset from raw Wiktionary data.
    
    This script:
    1. Streams raw JSONL entries from Kaikki.org (Wiktionary-derived data).
    2. Filters for English words with structured etymological templates.
    3. Merges duplicate entries (Wiktionary often has multiple senses per word).
    4. Selects the top N languages to keep as labels based on frequency.
    5. Remaps infrequent languages to the 'Other' category.
    6. Saves the compact records and labels to the 'data/' directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    args = parser.parse_args()
    config = load_config(args.config)
    
    root = Path(config["_project_root"])
    
    stats = ExtractionStats()
    records = []
    
    # Stream the multi-gigabyte Kaikki dump to avoid memory exhaustion
    print(f"Streaming data from {config['kaikki_url']}...")
    for entry in load_jsonl_from_url(config["kaikki_url"], config["max_source_lines"]):
        stats.source_lines += 1
        if entry.get("lang_code") == "en":
            stats.english_entries += 1
            
        # Parse the structured etymology from the raw Wiktionary entry
        record = compact_record(entry)
        if not record:
            stats.skipped_no_etymology += 1
            continue
            
        records.append(record)
        if config["max_records"] is not None and len(records) >= config["max_records"]:
            break
            
    # Combine entries for the same word (e.g., if a word has multiple etymological paths)
    merged = merge_records(records)
    
    # Analyze the distribution to determine which languages meet our inclusion threshold (default 1%)
    labels = select_labels(merged, config.get("top_n_languages", 0.01))
    
    # Map minority languages to 'Other'
    remapped = [remap_to_top_languages(record, labels["source_languages"]) for record in merged]
    
    # Save processed data
    records_path = resolve(config["_project_root"], config["records_path"])
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as f:
        for r in remapped:
            f.write(json.dumps(r) + "\n")
            
    labels_path = resolve(config["_project_root"], config["labels_path"])
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    
    # Track statistics for the dataset card
    stats_path = resolve(config["_project_root"], config["data_dir"]) / "extraction_stats.json"
    stats.records = len(remapped)
    stats_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
    
    print(json.dumps({"records": stats.records, "stats_path": str(stats_path)}, indent=2))

if __name__ == "__main__":
    main()

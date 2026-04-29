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
    write_jsonl,
    write_labels,
)
from etymology_tagger.storage import assert_under_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["_project_root"])
    assert_under_budget(root, config["storage_budget_gb"])
    stats = ExtractionStats()
    records = []
    for entry in load_jsonl_from_url(config["kaikki_url"], config["max_source_lines"]):
        stats.source_lines += 1
        if entry.get("lang_code") == "en":
            stats.english_entries += 1
        record = compact_record(entry)
        if not record:
            stats.skipped_no_etymology += 1
            continue
        records.append(record)
        if len(records) >= config["max_records"]:
            break
    merged = merge_records(records)
    labels = select_labels(merged, config["top_n_languages"])
    remapped = [remap_to_top_languages(record, labels["source_languages"]) for record in merged]
    stats.records = write_jsonl(resolve(config["_project_root"], config["records_path"]), remapped)
    write_labels(resolve(config["_project_root"], config["labels_path"]), labels)
    stats_path = resolve(config["_project_root"], config["data_dir"]) / "extraction_stats.json"
    stats_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")
    assert_under_budget(root, config["storage_budget_gb"])
    print(json.dumps({"records": stats.records, "stats_path": str(stats_path)}, indent=2))


if __name__ == "__main__":
    main()

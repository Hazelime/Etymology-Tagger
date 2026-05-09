from __future__ import annotations

import argparse
import json

import numpy as np

from common import load_config, resolve
from etymology_tagger.extract import read_jsonl
from etymology_tagger.model import (
    featurize,
    load_model,
    precision_recall_f1,
    predict_scores,
    split_indices,
)


def _threshold_array(value, fallback: float):
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)[None, :]
    if value is None:
        return fallback
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    args = parser.parse_args()
    config = load_config(args.config)
    records = read_jsonl(resolve(config["_project_root"], config["records_path"]))
    labels = json.loads(resolve(config["_project_root"], config["labels_path"]).read_text(encoding="utf-8"))
    metadata_path = resolve(config["_project_root"], config["metadata_path"])
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        thresholds = metadata.get("thresholds", {})
    else:
        thresholds = {}
    dataset = featurize(
        records,
        resolve(config["_project_root"], config["vector_subset_path"]),
        labels,
        config.get("orthographic_features"),
    )
    _, test_indices = split_indices(len(records), config["test_fraction"], config["random_seed"])
    model = load_model(resolve(config["_project_root"], config["model_path"]))
    metrics = {
        "source_language": precision_recall_f1(
            dataset.y_languages[test_indices],
            predict_scores(dataset.x[test_indices], model["language"][0], model["language"][1]),
            _threshold_array(thresholds.get("source_language"), config["threshold"]),
        ),
        "source_mechanism": precision_recall_f1(
            dataset.y_mechanisms[test_indices],
            predict_scores(dataset.x[test_indices], model["mechanism"][0], model["mechanism"][1]),
            _threshold_array(thresholds.get("source_mechanism"), config["threshold"]),
        ),
        "test_size": int(len(test_indices)),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from common import load_config, resolve
from etymology_tagger.extract import read_jsonl
from etymology_tagger.model import (
    featurize,
    precision_recall_f1,
    predict_scores,
    save_model,
    split_indices,
    train_two_head_model,
    tune_threshold,
    tune_thresholds_per_label,
    write_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    args = parser.parse_args()
    config = load_config(args.config)
    records = read_jsonl(resolve(config["_project_root"], config["records_path"]))
    labels = json.loads(resolve(config["_project_root"], config["labels_path"]).read_text(encoding="utf-8"))
    dataset = featurize(
        records,
        resolve(config["_project_root"], config["vector_subset_path"]),
        labels,
        config.get("orthographic_features"),
    )
    train_indices, test_indices = split_indices(
        len(records),
        config["test_fraction"],
        config["random_seed"],
    )
    model = train_two_head_model(dataset, train_indices, config)
    save_model(resolve(config["_project_root"], config["model_path"]), model)
    language_scores = predict_scores(
        dataset.x[test_indices],
        model["language"][0],
        model["language"][1],
    )
    language_train_scores = predict_scores(
        dataset.x[train_indices],
        model["language"][0],
        model["language"][1],
    )
    mechanism_scores = predict_scores(
        dataset.x[test_indices],
        model["mechanism"][0],
        model["mechanism"][1],
    )
    mechanism_train_scores = predict_scores(
        dataset.x[train_indices],
        model["mechanism"][0],
        model["mechanism"][1],
    )
    language_thresholds, language_train_metrics = tune_thresholds_per_label(
        dataset.y_languages[train_indices],
        language_train_scores,
    )
    mechanism_threshold, mechanism_train_metrics = tune_threshold(
        dataset.y_mechanisms[train_indices],
        mechanism_train_scores,
    )
    evaluation = {
        "source_language": precision_recall_f1(
            dataset.y_languages[test_indices],
            language_scores,
            language_thresholds[None, :],
        ),
        "source_mechanism": precision_recall_f1(
            dataset.y_mechanisms[test_indices],
            mechanism_scores,
            mechanism_threshold,
        ),
        "train_size": int(len(train_indices)),
        "test_size": int(len(test_indices)),
    }
    write_metadata(
        resolve(config["_project_root"], config["metadata_path"]),
        {
            "labels": labels,
            "thresholds": {
                "source_language": language_thresholds.tolist(),
                "source_mechanism": mechanism_threshold,
            },
            "orthographic_features": config.get("orthographic_features"),
            "train_threshold_metrics": {
                "source_language": language_train_metrics,
                "source_mechanism": mechanism_train_metrics,
            },
            "evaluation": evaluation,
            "model": "two independent weighted logistic heads over fastText word vectors plus hashed orthographic features",
        },
    )
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()

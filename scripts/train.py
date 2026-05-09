from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from common import load_config, resolve
from etymology_tagger.extract import read_jsonl
from etymology_tagger.model import (
    featurize,
    precision_recall_f1,
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
    
    # Undersampling Latin, English, French by 50% in the training set
    undersample_langs = ["Latin", "English", "French"]
    undersample_indices = [
        labels["source_languages"].index(l) 
        for l in undersample_langs if l in labels["source_languages"]
    ]
    if undersample_indices:
        rng = np.random.default_rng(config["random_seed"])
        has_major_lang = dataset.y_languages[train_indices][:, undersample_indices].sum(axis=1) > 0
        major_lang_train_indices = np.where(has_major_lang)[0]
        drop_count = len(major_lang_train_indices) // 2
        drop_indices_in_major = rng.choice(
            len(major_lang_train_indices), 
            size=drop_count, 
            replace=False
        )
        drop_indices = major_lang_train_indices[drop_indices_in_major]
        keep_mask = np.ones(len(train_indices), dtype=bool)
        keep_mask[drop_indices] = False
        train_indices = train_indices[keep_mask]
        print(f"Undersampled major languages: dropped {drop_count} samples. New train size: {len(train_indices)}")

    model = train_two_head_model(dataset, train_indices, config)
    save_model(resolve(config["_project_root"], config["model_path"]), model)
    
    with torch.no_grad():
        x_test_t = torch.tensor(dataset.x[test_indices], dtype=torch.float32)
        lang_logits_test, mech_logits_test = model(x_test_t)
        language_scores = torch.sigmoid(lang_logits_test).numpy()
        mechanism_scores = torch.sigmoid(mech_logits_test).numpy()
        
        x_train_t = torch.tensor(dataset.x[train_indices], dtype=torch.float32)
        lang_logits_train, mech_logits_train = model(x_train_t)
        language_train_scores = torch.sigmoid(lang_logits_train).numpy()
        mechanism_train_scores = torch.sigmoid(mech_logits_train).numpy()
    
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
            "model": "Y-Shape Multi-Task Learning MLP with Binary Focal Loss",
            "input_dim": dataset.x.shape[1]
        },
    )
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()

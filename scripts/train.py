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
)

def write_metadata(path: Path, metadata: dict) -> None:
    """Serializes training metadata and model thresholds to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def main() -> None:
    """
    Main training entry point for the Etymology Tagger.
    
    This script handles:
    1. Loading the dataset and pretrained word vectors.
    2. Featurizing words into hybrid semantic/orthographic vectors.
    3. Splitting data into train/test sets.
    4. Undersampling majority classes to improve model fairness.
    5. Training the Multi-Task Neural Network.
    6. Tuning decision thresholds on the training set.
    7. Evaluating on the held-out test set.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 1. Load data and labels
    records = list(read_jsonl(resolve(config["_project_root"], config["records_path"])))
    labels = json.loads(resolve(config["_project_root"], config["labels_path"]).read_text(encoding="utf-8"))
    
    # 2. Convert raw records into a vectorized Dataset
    dataset = featurize(
        records,
        resolve(config["_project_root"], config["vector_subset_path"]),
        labels,
        config.get("orthographic_features"),
    )
    
    # 3. Create reproducible train/test split
    train_indices, test_indices = split_indices(
        len(records),
        config["test_fraction"],
        config["random_seed"],
    )
    
    # 4. Undersampling Strategy
    # Latin, English, and French are extremely dominant in English etymology. 
    # To prevent the model from ignoring the long-tail of rare languages, we 
    # randomly drop 50% of samples containing these majority labels.
    undersample_langs = ["Latin", "English", "French"]
    undersample_indices = [
        labels["source_languages"].index(l) 
        for l in undersample_langs if l in labels["source_languages"]
    ]
    if undersample_indices:
        rng = np.random.default_rng(config["random_seed"])
        # Identify training samples that have at least one of the major languages
        has_major_lang = dataset.y_languages[train_indices][:, undersample_indices].sum(axis=1) > 0
        major_lang_train_indices = np.where(has_major_lang)[0]
        
        # Select 50% for removal
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

    # 5. Model Training (Y-Shape MTL)
    model = train_two_head_model(dataset, train_indices, config)
    save_model(resolve(config["_project_root"], config["model_path"]), model)
    
    # 6. Evaluation & Threshold Tuning
    # We tune thresholds on the training set to maximize F1, then evaluate on Test.
    with torch.no_grad():
        x_test_t = torch.tensor(dataset.x[test_indices], dtype=torch.float32)
        lang_logits_test, mech_logits_test = model(x_test_t)
        language_scores = torch.sigmoid(lang_logits_test).numpy()
        mechanism_scores = torch.sigmoid(mech_logits_test).numpy()
        
        x_train_t = torch.tensor(dataset.x[train_indices], dtype=torch.float32)
        lang_logits_train, mech_logits_train = model(x_train_t)
        language_train_scores = torch.sigmoid(lang_logits_train).numpy()
        mechanism_train_scores = torch.sigmoid(mech_logits_train).numpy()
    
    # Tune separate thresholds for each language (to handle sparse vs dense classes)
    language_thresholds, language_train_metrics = tune_thresholds_per_label(
        dataset.y_languages[train_indices],
        language_train_scores,
    )
    # Tune a single global threshold for entry mechanisms
    mechanism_threshold, mechanism_train_metrics = tune_threshold(
        dataset.y_mechanisms[train_indices],
        mechanism_train_scores,
    )
    
    # 7. Final Test Set Metrics
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
    
    # 8. Save Metadata
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

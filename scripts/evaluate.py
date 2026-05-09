from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import load_config, resolve
from etymology_tagger.extract import read_jsonl
from etymology_tagger.model import (
    featurize,
    load_model,
    precision_recall_f1,
    split_indices,
)

def _threshold_array(value, fallback: float):
    """Converts a scalar or list threshold into a NumPy array for vectorized comparison."""
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)[None, :]
    if value is None:
        return fallback
    return value

def main() -> None:
    """
    Stand-alone evaluation script to verify model performance on the test set.
    
    This script loads a trained PyTorch model and the associated metadata 
    (including tuned thresholds) to compute final Precision, Recall, and F1 scores.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.json")
    args = parser.parse_args()
    config = load_config(args.config)
    
    # 1. Load Data & Metadata
    records = list(read_jsonl(resolve(config["_project_root"], config["records_path"])))
    labels = json.loads(resolve(config["_project_root"], config["labels_path"]).read_text(encoding="utf-8"))
    metadata_path = resolve(config["_project_root"], config["metadata_path"])
    
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        thresholds = metadata.get("thresholds", {})
        input_dim = metadata.get("input_dim", 1324)
    else:
        print("Warning: No metadata found. Using default thresholds.")
        thresholds = {}
        input_dim = 1324
        
    # 2. Prepare Test Set
    dataset = featurize(
        records,
        resolve(config["_project_root"], config["vector_subset_path"]),
        labels,
        config.get("orthographic_features"),
    )
    
    # Ensure we use the exact same split as during training
    _, test_indices = split_indices(len(records), config["test_fraction"], config["random_seed"])
    
    # 3. Load PyTorch Model
    num_languages = len(labels["source_languages"])
    num_mechanisms = len(labels["mechanisms"])
    model = load_model(
        resolve(config["_project_root"], config["model_path"]),
        input_dim,
        num_languages,
        num_mechanisms
    )
    
    # 4. Inference
    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(dataset.x[test_indices], dtype=torch.float32)
        lang_logits, mech_logits = model(x_test_t)
        language_scores = torch.sigmoid(lang_logits).numpy()
        mechanism_scores = torch.sigmoid(mech_logits).numpy()
        
    # 5. Compute Metrics
    metrics = {
        "source_language": precision_recall_f1(
            dataset.y_languages[test_indices],
            language_scores,
            _threshold_array(thresholds.get("source_language"), 0.5),
        ),
        "source_mechanism": precision_recall_f1(
            dataset.y_mechanisms[test_indices],
            mechanism_scores,
            _threshold_array(thresholds.get("source_mechanism"), 0.5),
        ),
        "test_size": int(len(test_indices)),
    }
    
    print("Final Test Set Evaluation:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

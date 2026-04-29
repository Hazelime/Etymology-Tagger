from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import append_orthographic_features
from .vectors import load_vectors, vector_for_word


@dataclass
class Dataset:
    words: list[str]
    x: np.ndarray
    y_languages: np.ndarray
    y_mechanisms: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def labels_to_matrix(records: list[dict], labels: list[str], field: str) -> np.ndarray:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(records), len(labels)), dtype=np.float32)
    for row, record in enumerate(records):
        for label in record[field]:
            if label in index:
                matrix[row, index[label]] = 1.0
    return matrix


def featurize(records: list[dict], vector_path: Path, labels: dict, feature_config: dict | None = None) -> Dataset:
    word_to_index, vectors = load_vectors(vector_path)
    words = [record["word"] for record in records]
    x = np.vstack([vector_for_word(word, word_to_index, vectors) for word in words])
    x = append_orthographic_features(x, words, feature_config)
    return Dataset(
        words=words,
        x=x.astype(np.float32),
        y_languages=labels_to_matrix(records, labels["source_languages"], "source_languages"),
        y_mechanisms=labels_to_matrix(records, labels["mechanisms"], "mechanisms"),
    )


def train_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples, n_features = x_train.shape
    n_labels = y_train.shape[1]
    weights = rng.normal(0.0, 0.01, size=(n_features, n_labels)).astype(np.float32)
    bias = np.zeros(n_labels, dtype=np.float32)
    positives = y_train.sum(axis=0)
    negatives = n_samples - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 30.0).astype(np.float32)
    sample_weights = np.where(y_train > 0, pos_weight[None, :], 1.0).astype(np.float32)
    for _ in range(epochs):
        probabilities = sigmoid(x_train @ weights + bias)
        error = (probabilities - y_train) * sample_weights
        grad_w = (x_train.T @ error) / n_samples + l2 * weights
        grad_b = error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


def train_two_head_model(
    dataset: Dataset,
    train_indices: np.ndarray,
    config: dict,
) -> dict:
    x_train = dataset.x[train_indices]
    return {
        "language": train_head(
            x_train,
            dataset.y_languages[train_indices],
            config["epochs"],
            config["learning_rate"],
            config["l2"],
            config["random_seed"],
        ),
        "mechanism": train_head(
            x_train,
            dataset.y_mechanisms[train_indices],
            config["epochs"],
            config["learning_rate"],
            config["l2"],
            config["random_seed"] + 1,
        ),
    }


def predict_scores(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return sigmoid(x @ weights + bias)


def split_indices(n_items: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    test_size = max(1, int(round(n_items * test_fraction)))
    return indices[test_size:], indices[:test_size]


def precision_recall_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float | np.ndarray) -> dict:
    y_pred = (y_score >= threshold).astype(np.float32)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = precision_recall_f1(y_true, y_score, best_threshold)
    for threshold in np.linspace(0.1, 0.9, 33):
        metrics = precision_recall_f1(y_true, y_score, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def tune_thresholds_per_label(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, dict]:
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    for label_idx in range(y_true.shape[1]):
        threshold, _ = tune_threshold(
            y_true[:, label_idx : label_idx + 1],
            y_score[:, label_idx : label_idx + 1],
        )
        thresholds[label_idx] = threshold
    return thresholds, precision_recall_f1(y_true, y_score, thresholds[None, :])


def save_model(path: Path, model: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        payload = {
            "language_weights": model["language"][0].tolist(),
            "language_bias": model["language"][1].tolist(),
            "mechanism_weights": model["mechanism"][0].tolist(),
            "mechanism_bias": model["mechanism"][1].tolist(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return
    np.savez(
        path,
        language_weights=model["language"][0],
        language_bias=model["language"][1],
        mechanism_weights=model["mechanism"][0],
        mechanism_bias=model["mechanism"][1],
    )


def load_model(path: Path) -> dict:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "language": (
                np.asarray(payload["language_weights"], dtype=np.float32),
                np.asarray(payload["language_bias"], dtype=np.float32),
            ),
            "mechanism": (
                np.asarray(payload["mechanism_weights"], dtype=np.float32),
                np.asarray(payload["mechanism_bias"], dtype=np.float32),
            ),
        }
    data = np.load(path)
    return {
        "language": (data["language_weights"], data["language_bias"]),
        "mechanism": (data["mechanism_weights"], data["mechanism_bias"]),
    }


def write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

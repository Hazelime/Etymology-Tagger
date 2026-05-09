from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .features import append_orthographic_features
from .vectors import load_vectors, vector_for_word

@dataclass
class Dataset:
    """
    A structured container for processed etymology data.
    
    Attributes:
        words: List of word strings.
        x: Feature matrix (semantic + orthographic).
        y_languages: Binary matrix of source language labels.
        y_mechanisms: Binary matrix of entry mechanism labels.
    """
    words: list[str]
    x: np.ndarray
    y_languages: np.ndarray
    y_mechanisms: np.ndarray

def labels_to_matrix(records: list[dict], labels: list[str], field: str) -> np.ndarray:
    """
    Converts a list of labels in a record field into a multi-label binary matrix (One-Hot Encoding).
    
    This is necessary because words often have multiple etymological origins (e.g., a word 
    borrowed from French that ultimately traces back to Latin).
    """
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(records), len(labels)), dtype=np.float32)
    for row, record in enumerate(records):
        for label in record[field]:
            if label in index:
                matrix[row, index[label]] = 1.0
    return matrix

def featurize(records: list[dict], vector_path: Path, labels: dict, feature_config: dict | None = None) -> Dataset:
    """
    Transforms raw etymology records into a vectorized Dataset.
    
    Combines semantic fastText vectors (pretrained) with orthographic hashed features 
    (learned from character patterns).
    """
    word_to_index, vectors = load_vectors(vector_path)
    words = [record["word"] for record in records]
    
    # 1. Load semantic vectors
    x = np.vstack([vector_for_word(word, word_to_index, vectors) for word in words])
    
    # 2. Append orthographic features (character n-grams)
    x = append_orthographic_features(x, words, feature_config)
    
    return Dataset(
        words=words,
        x=x.astype(np.float32),
        y_languages=labels_to_matrix(records, labels["source_languages"], "source_languages"),
        y_mechanisms=labels_to_matrix(records, labels["mechanisms"], "mechanisms"),
    )

class Y_MTL_Model(nn.Module):
    """
    A Y-Shape Multi-Task Learning (MTL) Neural Network.
    
    The architecture consists of a 'Shared Trunk' that learns general linguistic features,
    branching into two specialized 'Heads' for different classification tasks.
    
    Benefits:
    - Shared features: The model learns that certain character patterns (like prefixes)
      signal both a specific language AND a borrowing process.
    - Efficiency: Training one model for two tasks is faster and requires less memory.
    """
    def __init__(self, input_dim: int, num_languages: int, num_mechanisms: int):
        super().__init__()
        # Shared trunk: Extract high-level features from the raw input
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4), # Prevent overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Source language head: Predicts linguistic origin families
        self.lang_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_languages)
        )
        
        # Entry mechanism head: Predicts the process (e.g., borrowed, inherited)
        # Note: This head is shallower as the task is lower complexity than language ID.
        self.mech_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, num_mechanisms)
        )

    def forward(self, x):
        shared_features = self.shared(x)
        lang_logits = self.lang_head(shared_features)
        mech_logits = self.mech_head(shared_features)
        return lang_logits, mech_logits

class BinaryFocalLoss(nn.Module):
    """
    Implementation of Binary Focal Loss to address severe class imbalance.
    
    Standard Cross Entropy ignores rare classes because it focuses on 'easy' majority classes.
    Focal Loss uses a 'focusing parameter' (gamma) to down-weight easy examples, forcing
    the model to focus on 'hard' minority examples (e.g., rare languages).
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets, pos_weight=None, mask=None):
        # 1. Calculate standard Binary Cross Entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. Apply the focal focusing modulating factor: (1 - pt)^gamma
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        # 3. Apply optional positive class weighting (Inverse Frequency)
        if pos_weight is not None:
            weights = torch.where(targets == 1, pos_weight, torch.ones_like(pos_weight))
            focal_loss = focal_loss * weights
            
        # 4. Apply masking for untagged samples (ignore samples with no label for this head)
        if mask is not None:
            focal_loss = focal_loss * mask.unsqueeze(1)
            return focal_loss.sum() / torch.clamp(mask.sum() * targets.size(1), min=1.0)
            
        return focal_loss.mean()

def train_two_head_model(
    dataset: Dataset,
    train_indices: np.ndarray,
    config: dict,
) -> nn.Module:
    """
    Main training loop for the Multi-Task model.
    
    Implements combined task loss weighting and gradient optimization.
    """
    x_train = torch.tensor(dataset.x[train_indices], dtype=torch.float32)
    y_lang_train = torch.tensor(dataset.y_languages[train_indices], dtype=torch.float32)
    y_mech_train = torch.tensor(dataset.y_mechanisms[train_indices], dtype=torch.float32)

    # Masks: Some words only have language tags, others only mechanism tags.
    # Masking ensures we don't penalize the model for missing info in the training data.
    lang_mask = (y_lang_train.sum(dim=1) > 0).float()
    mech_mask = (y_mech_train.sum(dim=1) > 0).float()

    # Inverse frequency weights: Calculate N / N_pos to boost signal for rare classes.
    n_samples = float(x_train.size(0))
    lang_positives = torch.clamp(y_lang_train.sum(dim=0), min=1.0)
    lang_pos_weight = torch.clamp(n_samples / lang_positives, min=1.0, max=10.0)
    
    mech_positives = torch.clamp(y_mech_train.sum(dim=0), min=1.0)
    mech_pos_weight = torch.clamp(n_samples / mech_positives, min=1.0, max=10.0)

    train_data = TensorDataset(x_train, y_lang_train, y_mech_train, lang_mask, mech_mask)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

    input_dim = x_train.size(1)
    num_languages = y_lang_train.size(1)
    num_mechanisms = y_mech_train.size(1)

    model = Y_MTL_Model(input_dim, num_languages, num_mechanisms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 0.001), weight_decay=config.get("l2", 0.01))
    criterion = BinaryFocalLoss(gamma=2.0)
    
    epochs = config.get("epochs", 50)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y_lang, batch_y_mech, batch_lang_mask, batch_mech_mask in train_loader:
            optimizer.zero_grad()
            lang_logits, mech_logits = model(batch_x)
            
            # Calculate individual task losses
            loss_lang = criterion(lang_logits, batch_y_lang, pos_weight=lang_pos_weight, mask=batch_lang_mask)
            loss_mech = criterion(mech_logits, batch_y_mech, pos_weight=mech_pos_weight, mask=batch_mech_mask)
            
            # Combine losses: Language task is prioritized (1.0 weight) over Mechanism (0.5)
            loss = 1.0 * loss_lang + 0.5 * loss_mech
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
            
    model.eval()
    return model

def split_indices(n_items: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper to perform reproducible train/test splits."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    test_size = max(1, int(round(n_items * test_fraction)))
    return indices[test_size:], indices[:test_size]

def precision_recall_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float | np.ndarray) -> dict:
    """Calculates standard IR metrics for evaluation."""
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
    """Optimizes the decision threshold to maximize F1 score."""
    best_threshold = 0.5
    best_metrics = precision_recall_f1(y_true, y_score, best_threshold)
    for threshold in np.linspace(0.1, 0.9, 33):
        metrics = precision_recall_f1(y_true, y_score, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics

def tune_thresholds_per_label(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, dict]:
    """Tunes separate thresholds for every class to handle different baseline frequencies."""
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    for label_idx in range(y_true.shape[1]):
        threshold, _ = tune_threshold(
            y_true[:, label_idx : label_idx + 1],
            y_score[:, label_idx : label_idx + 1],
        )
        thresholds[label_idx] = threshold
    return thresholds, precision_recall_f1(y_true, y_score, thresholds[None, :])

def save_model(path: Path, model: nn.Module) -> None:
    """Serializes the PyTorch model weights to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(path: Path, input_dim: int, num_languages: int, num_mechanisms: int) -> nn.Module:
    """Instantiates and loads the model from disk."""
    model = Y_MTL_Model(input_dim, num_languages, num_mechanisms)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model

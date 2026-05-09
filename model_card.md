---
license: cc-by-sa-4.0
language:
- en
library_name: torch
tags:
- etymology
- wiktionary
- fasttext
- multilabel-classification
- pytorch
- multi-task-learning
---

This predicts English word etymology across two separate dimensions using a **Y-Shape Multi-Task Learning (MTL) Neural Network** architecture:
1.  **Source Language Head**: Predicts the linguistic origin (e.g., Latin, Old Norse).
2.  **Entry Mechanism Head**: Predicts the etymological process (e.g., borrowing vs. inheritance).

### Model Architecture & Training

*   **Architecture**: The model features a shared MLP trunk (Input -> 512 -> 256, ReLU, Dropout 0.4) that splits into two independent task heads. This allows the model to learn a unified feature representation while optimizing for distinct etymological dimensions.
*   **Implementation**: Built with **PyTorch**. The move from logistic regression to a deep MTL architecture allows for capturing complex, non-linear dependencies between semantic and orthographic features.
*   **Feature Integration**:
    *   **fastText**: 300d pretrained vectors provide a semantic foundation.
    *   **Orthographic Hashing**: 1024-dimensional hashed features for character n-grams, prefixes, and suffixes capture sub-word morphology.
*   **Loss Optimization**:
    *   **Binary Focal Loss**: Trained with $\gamma=2.0$ to combat severe class imbalance. Focal loss reduces the loss contribution from easy-to-classify "major" languages and increases the importance of rare or ambiguous classes.
    *   **Multi-Task Weighting**: The total loss is balanced as $1.0 \times L_{language} + 0.5 \times L_{mechanism}$.
    *   **Inverse Frequency Weighting**: Positive samples are weighted by $N/N_{pos}$ (capped at 10.0) to ensure rare mechanisms like `calqued` are adequately learned.
    *   **Sample Masking**: Training ignores samples that do not have at least one identified label for the respective head, focusing the model on high-confidence data.

## Intended Use

This is intended as a working baseline and demo model, not a production-grade
historical linguistics resource. It is useful for exploring the task, validating the
pipeline, and powering the accompanying HuggingFace Space.

## Limitations

Etymology is layered and often ambiguous. A word can be inherited through one path,
borrowed through another, and ultimately derived from a third language. The demo
therefore displays paired labels from the source data when available, while the model
predicts the language and mechanism dimensions separately.

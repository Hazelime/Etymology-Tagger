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

# English Etymology Tagger

This model predicts English word etymology across two separate dimensions using a **Y-Shape Multi-Task Learning (MTL) Neural Network** architecture:

1.  **Source Language Classifier**: Predicts the linguistic origin (e.g., Latin, Old Norse).
2.  **Entry Mechanism Classifier**: Predicts the etymological process (e.g., borrowing vs. inheritance).

### Model Architecture & Training

*   **Architecture**: The model features a shared MLP trunk (Input -> 512 -> 256, ReLU, Dropout 0.4) that splits into two independent task heads. The **Language Head** utilizes a 128-neuron hidden layer for identifying origins across 23 families, while the **Mechanism Head** uses a 32-neuron layer for 4 entry processes.
*   **Implementation**: Built with **PyTorch**. The MTL architecture captures complex dependencies between semantic and orthographic features.
*   **Feature Integration**:
    *   **fastText**: We use the **wiki-news-300d-1M** model (300 dimensions), trained on Wikipedia 2017, Statmt.org news, and the UMBC corpus.
    *   **Orthographic Hashing**: 1024-dimensional hashed features for character n-grams, prefixes, and suffixes capture sub-word morphology.
*   **Loss Optimization**:
    *   **Binary Focal Loss**: Trained with $\gamma=2.0$ to combat severe class imbalance by focusing on "hard" examples.
    *   **Multi-Task Weighting**: The total loss is balanced as $1.0 \times L_{language} + 0.5 \times L_{mechanism}$.
    *   **Inverse Frequency Weighting**: Positive samples are weighted by $N/N_{pos}$ (capped at 10.0), where $N$ is the total training count and $N_{pos}$ is the count for a specific class.
    *   **Sample Masking**: Gradients are masked for samples lacking valid labels for a specific task.

## Intended Use
This system is designed for automated etymological research and as a baseline for computational historical linguistics. It powers the accompanying interactive demo.

## Limitations
Etymology is layered and often ambiguous. A word can be inherited through one path, borrowed through another, and ultimately derived from a third language. The model predicts the primary language and mechanism dimensions separately, representing the paths captured in structured Wiktionary data.

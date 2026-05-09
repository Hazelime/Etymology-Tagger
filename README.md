# English Etymology Tagger

A neural etymology tagger produced as a course assignment for **Information Retrieval (5LN712)** at Uppsala University.

This system provides automated etymological analysis for English words, utilizing Wiktionary-derived data from [Kaikki](https://kaikki.org/), pretrained fastText word vectors, and high-dimensional orthographic features.

## Architecture: Y-Shape Multi-Task Learning (MTL)

The tagger uses a **Y-Shape Multi-Task Learning (MTL) Neural Network** architecture designed to jointly model the overlapping dimensions of etymological origin:

1.  **Shared Trunk**: A deep MLP (512 -> 256 neurons) with ReLU activation and Dropout (0.4) that extracts a unified representation from semantic and orthographic features.
2.  **Source Language Head**: A task-specific head predicting the most likely linguistic origins (e.g., `Latin`, `Proto-Germanic`).
3.  **Entry Mechanism Head**: A task-specific head predicting the primary etymological process (e.g., `borrowed`, `derived`, `inherited`).

## Technical Design & Motivation

*   **Multi-Task Approach**: Etymological mechanisms (the process of entry) and source languages are deeply intertwined. The Y-shape architecture allows the model to learn features useful for both tasks in the shared trunk, while the dual heads enable independent decision boundaries for each dimension.
*   **Dual Classifiers**: The system utilizes two distinct classification heads branching from the shared trunk:
    *   **Source Language Classifier**: A dense head with a 128-neuron hidden layer optimized for multi-label classification across 23 language families.
    *   **Entry Mechanism Classifier**: A specialized head with a 32-neuron hidden layer focused on 4 entry processes. This smaller capacity reflects the lower complexity of the mechanism task compared to language identification.
*   **Binary Focal Loss**: The model is optimized using **Binary Focal Loss** ($\gamma=2.0$). This addresses extreme class imbalance by down-weighting the loss contribution from well-classified "easy" examples and focusing gradient updates on "hard" minority classes (e.g., rare languages).
*   **Embeddings & Features**:
    *   **Semantic**: We utilize the **wiki-news-300d-1M** fastText model. These 300-dimensional vectors were trained with subword information on **Wikipedia 2017**, **Statmt.org news**, and the **UMBC corpus**, providing a robust semantic foundation.
    *   **Orthographic**: 1024-dimensional hashed features derived from character n-grams (3-5), prefixes, and suffixes capture morphological signals (e.g., the `-ation` suffix suggesting a Latinate origin).
*   **Addressing Class Imbalance**:
    *   **Undersampling**: High-frequency classes (Latin, English, French) are randomly undersampled by 50% during training.
    *   **Inverse Frequency Weighting**: Initial class weights are calculated as $N / N_{pos}$ (capped at 10.0), where $N$ is the total number of training samples and $N_{pos}$ is the number of positive samples for a specific class.
    *   **Sample Masking**: The model ignores datapoints lacking valid tags for a specific head, ensuring gradients only reflect confirmed information.

## Getting Started

### 1. Training & Evaluation
To rebuild the dataset and train the model from scratch:

```powershell
$PY="python" # Or path to your Python executable
& $PY scripts/build_dataset.py --config configs/prototype.json
& $PY scripts/prepare_fasttext_subset.py --config configs/prototype.json
& $PY scripts/train.py --config configs/prototype.json
& $PY scripts/evaluate.py --config configs/prototype.json
```

### 2. Running the Demo
The system includes an interactive Gradio demo (`app.py`).

```powershell
pip install -r requirements.txt
python app.py
```

## Data & Storage
The dataset builder streams Kaikki data and writes only compact etymology records to disk. The vector preparer downloads the official fastText archive temporarily, extracts only the required subset, and cleans up to respect a configurable storage budget (default 5 GB).

## Outputs
- `data/processed/etymology_records.jsonl`: Compact Wiktionary-derived records.
- `data/processed/labels.json`: Language and mechanism label sets.
- `models/fasttext_subset.vec`: 300d pretrained vector subset.
- `models/etymology_tagger.pt`: Trained PyTorch MTL model.
- `models/metadata.json`: Thresholds, labels, and evaluation metrics.

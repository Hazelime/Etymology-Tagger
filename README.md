# English Etymology Tagger

A vibe-coded etymology tagger produced as a course assignment for **Information Retrieval (5LN712)** at Uppsala University.

Prototype word etymology tagger for English, using Wiktionary-derived JSONL from
[Kaikki](https://kaikki.org/), fastText-style pretrained word vectors, and hashed
orthographic features.

The tagger uses a **Y-Shape Multi-Task Learning (MTL) Neural Network** architecture designed to model the overlapping nature of etymological origins:
1. **Shared Trunk**: A deep MLP (512 -> 256 neurons) with ReLU activation and Dropout (0.4) extracts a unified representation from semantic and orthographic features.
2. **Source Language Head**: A task-specific head predicting the likely source languages (e.g., `Latin`, `Proto-Germanic`).
3. **Entry Mechanism Head**: A task-specific head predicting the primary etymological process (e.g., `borrowed`, `derived`, `inherited`).

### Technical Design & Motivation

*   **Why Y-Shape MTL?** Etymological mechanisms (how a word entered the language) and source languages are deeply intertwined. A shared trunk allows the model to learn features that are useful for both tasks (e.g., certain character n-grams that signal both a specific language and a borrowing process), while task-specific heads allow for independent decision boundaries.
*   **Classification Model**: The system uses a **Deep Neural Network** implemented in **PyTorch**. This replaced the legacy logistic regression cascade to better capture non-linear interactions between orthographic and semantic features.
*   **Loss Function (Binary Focal Loss)**:
    *   The model is trained using **Binary Focal Loss** ($\gamma=2.0$). This addresses extreme class imbalance by down-weighting well-classified "easy" examples and focusing the gradient update on "hard" minority classes.
    *   **Combined Loss**: The total loss is $L_{total} = 1.0 \times L_{language} + 0.5 \times L_{mechanism}$, prioritizing language accuracy while maintaining a strong signal for entry mechanisms.
*   **Feature Engineering**:
    *   **Semantic**: Frozen **fastText** word vectors capture semantic context.
    *   **Orthographic**: Hashed character n-grams (1024-d), prefixes, and suffixes capture morphological cues (e.g., the `-ation` suffix suggesting a Latin/French origin).
*   **Addressing Class Imbalance**:
    *   **Undersampling**: To prevent the model from being dominated by high-frequency classes, we randomly undersample **Latin**, **English**, and **French** datapoints by 50% during training.
    *   **Inverse Frequency Weighting**: Initial class weights are calculated as $N / N_{pos}$ (capped at 10.0) to further balance the loss signal across rare labels.
    *   **Sample Masking**: The model ignores datapoints that lack valid tags for a specific head, ensuring gradients only reflect confirmed etymological information.
*   **Linguistic Consolidation**: Common language variants are collapsed into parent families (e.g., Old/Middle English variants merge into `English`) to improve precision and reduce class sparsity.

## Storage Behavior

The dataset builder streams Kaikki data and writes only compact etymology records to disk.
It does not save the full Wiktionary dump. The vector preparer downloads the official
fastText zip temporarily, extracts only vectors needed by the compact dataset, then deletes
the zip. Both scripts enforce a configurable storage budget, defaulting to 5 GB.

## Quick Prototype

Use the bundled Python executable if `python` is not on PATH.

```powershell
$PY="C:\Users\marcu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY scripts/build_dataset.py --config configs/prototype.json
& $PY scripts/prepare_fasttext_subset.py --config configs/prototype.json
& $PY scripts/train.py --config configs/prototype.json
& $PY scripts/evaluate.py --config configs/prototype.json
```

For a very small smoke test that avoids the fastText download, use:

```powershell
$PY="C:\Users\marcu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY scripts/build_tiny_demo_assets.py --config configs/prototype.json
& $PY scripts/evaluate.py --config configs/prototype.json
```

The smoke test exists only to verify the project wiring. For real results, run the
fastText subset command above.

## Demo

The HuggingFace Space entry point is `app.py`. It expects a trained model under
`models/` and compact etymology data under `data/processed/`.

Local launch, after installing dependencies:

```powershell
pip install -r requirements.txt
python app.py
```

## Outputs

- `data/processed/etymology_records.jsonl`: compact Wiktionary-derived records
- `data/processed/labels.json`: selected language and mechanism label sets
- `models/fasttext_subset.vec`: compact pretrained vector subset
- `models/etymology_tagger.pt`: trained PyTorch MTL model
- `models/metadata.json`: thresholds, labels, and evaluation metadata

## Later Upload Step

Once the local project is ready, upload separately:

- project repository to GitHub
- compact dataset to HuggingFace Datasets
- demo app to HuggingFace Spaces

Credentials and target repository URLs should be provided only when that step begins.

# English Etymology Tagger

Prototype word etymology tagger for English, using Wiktionary-derived JSONL from
[Kaikki](https://kaikki.org/), fastText-style pretrained word vectors, and hashed
orthographic features.

The tagger predicts two separate multi-label dimensions:

- source language, configurable as the top-N most common source languages plus `Other`
- source mechanism, currently `borrowed`, `derived`, `calqued`, and `inherited`

The dataset keeps language-mechanism pairs for presentation, so the demo can show labels
such as `borrowed from French` even though the model heads are trained separately.

The model feature layer combines frozen fastText vectors with lightweight prefix,
suffix, and character n-gram features. Two weighted logistic heads then predict the
source-language and source-mechanism labels.

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
- `models/etymology_tagger.json`: trained two-head model
- `models/metadata.json`: thresholds, labels, and evaluation metadata

## Later Upload Step

Once the local project is ready, upload separately:

- project repository to GitHub
- compact dataset to HuggingFace Datasets
- demo app to HuggingFace Spaces

Credentials and target repository URLs should be provided only when that step begins.

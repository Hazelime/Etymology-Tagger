---
title: English Etymology Tagger
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: cc-by-sa-4.0
---

# English Etymology Tagger

Interactive demo for a prototype English etymology tagger.

The model uses a **Y-Shape Multi-Task Learning (MTL) Neural Network** architecture:
1.  **Shared Trunk**: A deep MLP extracts unified features from semantic and orthographic inputs.
2.  **Source Language Head**: Identifies the linguistic origin (e.g., Latin, Old Norse).
3.  **Entry Mechanism Head**: Identifies the etymological path (e.g., borrowing vs. inheritance).

Features are derived from pretrained **fastText** vectors and hashed orthographic n-grams. The model is implemented in **PyTorch** and optimized using **Binary Focal Loss** to handle class imbalance across etymological categories.

The app includes a permanent **Model Performance** table at the bottom of the page, showing precision, recall, and F1 scores validated on a held-out test set.

The compact etymology records are derived from English Wiktionary data published by [Kaikki](https://kaikki.org/).

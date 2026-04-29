---
license: cc-by-sa-4.0
language:
- en
library_name: numpy
tags:
- etymology
- wiktionary
- fasttext
- multilabel-classification
---

# English Etymology Tagger Prototype

This prototype predicts English word etymology across two separate multi-label
dimensions:

- source language
- source mechanism

The model uses pretrained fastText word vectors as frozen input features, augmented
with hashed orthographic features for prefixes, suffixes, and character n-grams.
Two weighted logistic heads are trained on compact Wiktionary-derived etymology
records.

## Intended Use

This is intended as a working baseline and demo model, not a production-grade
historical linguistics resource. It is useful for exploring the task, validating the
pipeline, and powering the accompanying HuggingFace Space.

## Limitations

Etymology is layered and often ambiguous. A word can be inherited through one path,
borrowed through another, and ultimately derived from a third language. The demo
therefore displays paired labels from the source data when available, while the model
predicts the language and mechanism dimensions separately.

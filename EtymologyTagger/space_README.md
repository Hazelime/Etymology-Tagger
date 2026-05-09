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

Users can paste text, tag each word by its most probable source language, and click
individual words to see a fuller etymological breakdown with paired labels such as
`borrowed from French` or `inherited from Middle English`.

The model uses pretrained fastText vectors plus hashed prefix, suffix, and
character n-gram features, followed by two weighted multi-label classification
heads:

- source language
- source mechanism

The compact etymology records are derived from English Wiktionary data published by
[Kaikki](https://kaikki.org/).

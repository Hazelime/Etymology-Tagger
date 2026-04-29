---
license: cc-by-sa-4.0
language:
- en
task_categories:
- token-classification
- text-classification
pretty_name: English Etymology Tagger Prototype Dataset
---

# English Etymology Tagger Prototype Dataset

This is a compact prototype dataset derived from English entries in Wiktionary,
using machine-readable JSONL from [Kaikki](https://kaikki.org/).

The dataset stores only etymology-relevant fields:

- `word`
- `parts_of_speech`
- `source_languages`
- `mechanisms`
- `pairs`, preserving paired labels such as `borrowed from French`
- `etymology_texts`, for demo explanations

It intentionally does not include the full Kaikki/Wiktionary dump.

## Labels

Source language labels are selected as a configurable top-N set plus `Other`.
Mechanism labels are:

- `borrowed`
- `derived`
- `calqued`
- `inherited`

## Source

The source is English Wiktionary data extracted by Wiktextract and published by
Kaikki. If you use this dataset, cite Wiktionary, Kaikki, and Wiktextract.

## Limitations

This is a prototype dataset. Wiktionary etymology templates are rich, inconsistent,
and historically layered. Some words have multiple valid labels, and the labels here
are extracted from selected etymology templates rather than manually curated.

---
license: cc-by-sa-4.0
language:
- en
task_categories:
- token-classification
- text-classification
pretty_name: English Etymology Tagger Dataset
dataset_info:
  features:
  - name: word
    dtype: string
  - name: display_word
    dtype: string
  - name: parts_of_speech
    sequence: string
  - name: etymology_texts
    sequence: string
  - name: pairs
    list:
    - name: mechanism
      dtype: string
    - name: source_language
      dtype: string
    - name: source_code
      dtype: string
    - name: source_term
      dtype: string
    - name: template
      dtype: string
    - name: detail
      dtype: string
  - name: source_languages
    sequence: string
  - name: mechanisms
    sequence: string
  splits:
  - name: train
    num_examples: 102111
configs:
- config_name: default
  data_files:
  - split: train
    path: etymology_records.jsonl
---

# English Etymology Tagger Dataset

A refined dataset of English word etymologies derived from Wiktionary entries via machine-readable JSONL from [Kaikki](https://kaikki.org/).

## Dataset Statistics

### 1. Data Pipeline & Volume
- **Original Source**: 1,465,676 English entries (total lines in the Kaikki English JSONL).
- **Extracted Datapoints**: 102,111 entries (words containing etymological templates parsed by `wiktextract`).

### 2. Source Language Distribution (Top 10)
Based on the relevant datapoints ($N = 102,111$):

| Language | Frequency | Percentage |
| :--- | :--- | :--- |
| **Latin** (consolidated) | 23,728 | 23.24% |
| **English** (consolidated) | 20,974 | 20.54% |
| **French** (consolidated) | 16,694 | 16.35% |
| **Greek** (consolidated) | 10,074 | 9.87% |
| **German** (consolidated) | 7,300 | 7.15% |
| **Chinese** (consolidated) | 5,017 | 4.91% |
| **Italian** | 4,729 | 4.63% |
| **Proto-Germanic** (consolidated) | 4,692 | 4.59% |
| **Spanish** | 4,628 | 4.53% |
| **Proto-Indo-European** | 4,097 | 4.01% |
| ... | ... | ... |
| **Other** (all languages < 1%) | 30,339 | 29.71% |

### 3. Entry Mechanism Distribution
Based on the extracted datapoints ($N = 102,111$). The dataset categorizes origins into four primary mechanisms:
- **borrowed**: Words taken directly from another language (e.g., *sushi* from Japanese).
- **derived**: Words formed by morphological derivation from another language's roots.
- **inherited**: Words passed down continuously from an ancestor language (e.g., from Old English to Modern English).
- **calqued**: Word-for-word translations of phrases or compound words from another language (e.g., *skyscraper* translated into French as *gratte-ciel*).

| Mechanism | Frequency | Percentage |
| :--- | :--- | :--- |
| **borrowed** | 57,530 | 56.34% |
| **derived** | 45,760 | 44.81% |
| **inherited** | 18,032 | 17.66% |
| **calqued** | 2,231 | 2.18% |

*(Note: Percentages sum to >100% because words can have multiple labels. The "Other" category aggregates hundreds of minority languages.)*

### Data Processing & Refinement

The data provided in this dataset (`etymology_records.jsonl`) contains the **102,111 parsed records**. The following normalization is applied during the creation of this dataset:

1.  **Linguistic Consolidation**: Common historical and regional variants are mapped to their primary language families to reduce sparsity. The consolidated language families include: **Latin, English, French, German, Greek, Chinese, Dutch, Scots, and Proto-Germanic**.
2.  **Exclusions**: Non-etymological labels (e.g., "Translingual") or structurally empty templates are removed.

## Source
The source is English Wiktionary data extracted by Wiktextract and published by [Kaikki](https://kaikki.org/).

## Limitations
Etymology represents the primary paths captured in Wiktionary templates and may not reflect every historical nuance for complex terms.

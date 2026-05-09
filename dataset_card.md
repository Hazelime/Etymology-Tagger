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

This is a compact prototype dataset derived from English entries in Wiktionary, using machine-readable JSONL from [Kaikki](https://kaikki.org/).

## Dataset Statistics

### 1. Data Pipeline & Volume
- **Original Source**: 1,465,676 English entries (total lines in the Kaikki English JSONL).
- **Relevant Datapoints**: 102,111 entries (words containing etymological templates parsed by `wiktextract`).
- **Filtered & Processed**: 83,204 datapoints were used for model training and evaluation (after collapsing variants and ensuring at least one valid label remained).

### 2. Source Language Distribution (Top 10)
Based on the relevant datapoints ($N = 102,111$):

| Language | Frequency | Percentage |
| :--- | :--- | :--- |
| **Latin** (consolidated) | 23,728 | 23.24% |
| **English** (consolidated) | 20,974 | 20.54% |
| **French** (consolidated) | 16,694 | 16.35% |
| **Greek** | 10,074 | 9.87% |
| **German** | 7,300 | 7.15% |
| **Chinese** | 5,017 | 4.91% |
| **Italian** | 4,729 | 4.63% |
| **Proto-Germanic** | 4,692 | 4.59% |
| **Spanish** | 4,628 | 4.53% |
| **Proto-Indo-European** | 4,097 | 4.01% |
| ... | ... | ... |
| **Other** (all languages < 1%) | 30,339 | 29.71% |

### 3. Entry Mechanism Distribution
Based on the relevant datapoints ($N = 102,111$):

| Mechanism | Frequency | Percentage |
| :--- | :--- | :--- |
| **borrowed** | 57,530 | 56.34% |
| **derived** | 45,760 | 44.81% |
| **inherited** | 18,032 | 17.66% |
| **calqued** | 2,231 | 2.18% |

*(Note: Percentages sum to >100% because words can have multiple etymological paths/labels. The "Other" category is significant because it aggregates hundreds of rare languages including Ukrainian, Welsh, and Portuguese.)*

### Data Processing & Refinement
1.  **Linguistic Consolidation (Collapsing)**: Numerous Wiktionary language variants were mapped to their primary families to reduce sparsity.
    *   **Latin variants** (Late, Medieval, Vulgar, etc.) → **Latin**.
    *   **English variants** (Old, Middle, etc.) → **English**.
    *   **Germanic Merges**: "Proto-West Germanic" was merged into **"Proto-Germanic"** to provide a more stable training signal.
2.  **Label Thresholding**: Only languages appearing in **>1%** of the dataset (~1,021 occurrences) are preserved as distinct labels. All others are remapped to **"Other"**.
3.  **Exclusions**: Non-etymological labels such as **"Translingual"** and broad categories like **"Germanic languages"** were removed.
4.  **Imbalance Mitigation**:
    *   **Undersampling**: The training set implements a random **50% undersampling** for words originating from **Latin**, **English**, and **French**.
    *   **Multi-Task Optimization**: The dataset is designed for Multi-Task Learning (MTL), allowing models to learn shared representations for both source languages and entry mechanisms.

## Source
The source is English Wiktionary data extracted by Wiktextract and published by [Kaikki](https://kaikki.org/).

## Limitations
Etymology is often ambiguous or layered. A word might have an inherited root but be heavily influenced by a subsequent borrowing (e.g., "skirt" vs. "shirt"). This dataset represents the primary paths captured in Wiktionary templates.

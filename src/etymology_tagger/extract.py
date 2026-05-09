from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

from .languages import language_name

# Mapping of Wiktionary etymology templates to simplified model mechanisms.
# Wiktionary has dozens of specific templates (e.g., 'learned borrowing'), 
# which we collapse into four broad categories for more stable training.
MECHANISM_BY_TEMPLATE = {
    "bor": "borrowed",
    "bor+": "borrowed",
    "borrowed": "borrowed",
    "lbor": "borrowed",
    "lbor+": "borrowed",
    "learned borrowing": "borrowed",
    "obor": "borrowed",
    "slbor": "borrowed",
    "ubor": "borrowed",
    "der": "derived",
    "der+": "derived",
    "derived": "derived",
    "inh": "inherited",
    "inh+": "inherited",
    "inherited": "inherited",
    "calque": "calqued",
    "clq": "calqued",
    "cal": "calqued",
    "semantic calque": "calqued",
}

MECHANISMS = ["borrowed", "derived", "calqued", "inherited"]

# Validation regex for English headwords (allows apostrophes and hyphens).
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{0,48}$")

def collapse_language(name: str | None) -> str | None:
    """
    Consolidates fine-grained linguistic variants into primary family labels.
    
    Wiktionary distinguishes between 'Late Latin', 'Vulgar Latin', 'Medieval Latin', etc.
    For a general-purpose tagger, these often provide redundant or sparse signals.
    Collapsing them into 'Latin' ensures the model has enough samples per class.
    """
    if not name:
        return None
    name = name.strip()
    
    # Exclude broad or meta-categories that don't provide specific origin info.
    if name in ["Translingual", "Germanic languages"]:
        return None
        
    # Mapping table for consolidation
    collapsing = {
        "Latin": ["Late Latin", "Medieval Latin", "New Latin", "Vulgar Latin", "Classical Latin", "Ecclesiastical Latin", "la-ecc"],
        "English": ["Old English", "Middle English", "Northern Middle English", "ang", "enm", "enm-nor"],
        "Scots": ["Middle Scots", "gmw-msc"],
        "French": ["Old French", "Middle French", "Old Northern French", "Anglo-Norman", "fro", "frm", "fro-nor", "xno"],
        "German": ["Old High German", "Middle High German", "Alemannic German", "Low German", "Old Saxon", "Middle Low German", "goh", "gmh", "gsw", "nds", "osx", "gml"],
        "Greek": ["Ancient Greek", "Byzantine Greek", "Koine Greek", "grc", "gkm", "grc-koi"],
        "Chinese": ["Mandarin Chinese", "Cantonese", "Hokkien", "Teochew", "Middle Chinese", "Mandarin Chinese (Pinyin)", "Mandarin Chinese (Wade-Giles)", "Mandarin Chinese (Tongyong)", "cmn", "yue", "nan-hbl", "nan-tws", "ltc", "cmn-pinyin", "cmn-wadegiles", "cmn-tongyong"],
        "Dutch": ["Middle Dutch", "Old Dutch", "dum", "odt"],
        "Proto-Germanic": ["Proto-West Germanic", "gmw-pro"],
    }
    
    for parent, variants in collapsing.items():
        if name == parent or name in variants:
            return parent
            
    return name

@dataclass
class ExtractionStats:
    """Container for tracking data volume through the extraction pipeline."""
    source_lines: int = 0
    english_entries: int = 0
    records: int = 0
    skipped_no_etymology: int = 0

def load_jsonl_from_url(url: str, max_source_lines: int | None = None) -> Iterable[dict]:
    """Streams JSONL records from a URL (e.g., Kaikki.org) to avoid loading massive files into memory."""
    request = Request(url, headers={"User-Agent": "EtymologyTagger/0.1"})
    with urlopen(request, timeout=60) as response:
        for idx, raw_line in enumerate(response, start=1):
            if max_source_lines and idx > max_source_lines:
                break
            if not raw_line.strip():
                continue
            yield json.loads(raw_line)

def template_mechanism(name: str | None) -> str | None:
    """Extracts the mechanism label from a Wiktextract template name."""
    if not name:
        return None
    normalized = name.strip().lower()
    return MECHANISM_BY_TEMPLATE.get(normalized)

def source_from_template(template: dict) -> tuple[str | None, str | None]:
    """
    Parses a Wiktextract template to find the source language code and name.
    
    Handles cases where the first argument is English ('en') and the actual 
    source is the second argument.
    """
    args = template.get("args") or {}
    code = args.get("2")
    if code == "en":
        code = args.get("3")
    name = language_name(code)
    return code, collapse_language(name)

def extract_etymology(record: dict) -> dict | None:
    """
    Core parser for a single Wiktionary record (from Kaikki/Wiktextract).
    
    Extracts the word, etymology text, and all structured language/mechanism 
    pairs found in the templates.
    """
    word = record.get("word")
    if not word or not WORD_RE.match(word):
        return None
        
    lang_code = record.get("lang_code")
    if lang_code != "en":
        return None

    # We look for structured etymological templates parsed by Wiktextract
    etym_templates = record.get("etymology_templates", [])
    etym_texts = record.get("etymology_text", [])
    if isinstance(etym_texts, str):
        etym_texts = [etym_texts]
    
    pairs = []
    source_langs = set()
    mechanisms = set()
    
    for template in etym_templates:
        name = template.get("name")
        mech = template_mechanism(name)
        if not mech:
            continue
            
        code, lang = source_from_template(template)
        if not lang:
            continue
            
        pairs.append({"source_language": lang, "mechanism": mech, "code": code})
        source_langs.add(lang)
        mechanisms.add(mech)
        
    if not source_langs and not mechanisms:
        return None
        
    return {
        "word": word.lower(),
        "source_languages": list(source_langs),
        "mechanisms": list(mechanisms),
        "pairs": pairs,
        "etymology_texts": etym_texts,
    }

def read_jsonl(path: Path) -> Iterable[dict]:
    """Utility to read a local JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def select_labels(records: Iterable[dict], threshold_percent: float = 0.01) -> dict:
    """
    Analyzes the extracted dataset to select which labels to keep in the model.
    
    Any language with a frequency below the threshold_percent is remapped to 'Other'
    during training to prevent the model from learning from insufficient data.
    """
    language_counts = Counter()
    total_records = 0
    for record in records:
        total_records += 1
        language_counts.update(record["source_languages"])
        
    threshold = total_records * threshold_percent
    languages = [lang for lang, count in language_counts.most_common() if count >= threshold]
    if "Other" not in languages:
        languages.append("Other")
        
    return {
        "source_languages": languages,
        "mechanisms": MECHANISMS,
        "language_counts": dict(language_counts.most_common()),
    }

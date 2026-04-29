from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

from .languages import language_name

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
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{0,48}$")


@dataclass
class ExtractionStats:
    source_lines: int = 0
    english_entries: int = 0
    records: int = 0
    skipped_no_etymology: int = 0


def load_jsonl_from_url(url: str, max_source_lines: int | None = None) -> Iterable[dict]:
    request = Request(url, headers={"User-Agent": "EtymologyTagger/0.1"})
    with urlopen(request, timeout=60) as response:
        for idx, raw_line in enumerate(response, start=1):
            if max_source_lines and idx > max_source_lines:
                break
            if not raw_line.strip():
                continue
            yield json.loads(raw_line)


def template_mechanism(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.strip().lower()
    return MECHANISM_BY_TEMPLATE.get(normalized)


def source_from_template(template: dict) -> tuple[str | None, str | None]:
    args = template.get("args") or {}
    code = args.get("2")
    if code == "en":
        code = args.get("3")
    name = language_name(code)
    return code, name


def pair_from_template(template: dict) -> dict | None:
    mechanism = template_mechanism(template.get("name"))
    if not mechanism:
        return None
    source_code, source_language = source_from_template(template)
    if not source_language:
        return None
    args = template.get("args") or {}
    return {
        "mechanism": mechanism,
        "source_language": source_language,
        "source_code": source_code,
        "source_term": args.get("3") if args.get("2") != "en" else args.get("4"),
        "template": template.get("name"),
        "detail": template.get("expansion") or "",
    }


def compact_record(entry: dict) -> dict | None:
    if entry.get("lang_code") != "en":
        return None
    word = (entry.get("word") or "").strip()
    if not WORD_RE.match(word):
        return None
    templates = entry.get("etymology_templates") or []
    pairs = []
    seen = set()
    for template in templates:
        pair = pair_from_template(template)
        if not pair:
            continue
        key = (pair["mechanism"], pair["source_language"], pair.get("source_term"))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pair)
    if not pairs:
        return None
    return {
        "word": word.lower(),
        "display_word": word,
        "pos": entry.get("pos"),
        "etymology_text": entry.get("etymology_text") or "",
        "pairs": pairs,
        "source_languages": sorted({pair["source_language"] for pair in pairs}),
        "mechanisms": sorted({pair["mechanism"] for pair in pairs}),
    }


def merge_records(records: Iterable[dict]) -> list[dict]:
    by_word: dict[str, dict] = {}
    for record in records:
        word = record["word"]
        target = by_word.setdefault(
            word,
            {
                "word": word,
                "display_word": record.get("display_word", word),
                "parts_of_speech": [],
                "etymology_texts": [],
                "pairs": [],
                "source_languages": [],
                "mechanisms": [],
            },
        )
        if record.get("pos") and record["pos"] not in target["parts_of_speech"]:
            target["parts_of_speech"].append(record["pos"])
        if record.get("etymology_text") and record["etymology_text"] not in target["etymology_texts"]:
            target["etymology_texts"].append(record["etymology_text"])
        seen_pairs = {
            (p["mechanism"], p["source_language"], p.get("source_term"))
            for p in target["pairs"]
        }
        for pair in record["pairs"]:
            key = (pair["mechanism"], pair["source_language"], pair.get("source_term"))
            if key not in seen_pairs:
                target["pairs"].append(pair)
                seen_pairs.add(key)
        target["source_languages"] = sorted({p["source_language"] for p in target["pairs"]})
        target["mechanisms"] = sorted({p["mechanism"] for p in target["pairs"]})
    return list(by_word.values())


def select_labels(records: list[dict], top_n_languages: int) -> dict:
    language_counts = Counter(
        language for record in records for language in record["source_languages"]
    )
    languages = [language for language, _ in language_counts.most_common(top_n_languages)]
    if "Other" not in languages:
        languages.append("Other")
    return {
        "source_languages": languages,
        "mechanisms": MECHANISMS,
        "language_counts": dict(language_counts.most_common()),
    }


def remap_to_top_languages(record: dict, selected_languages: list[str]) -> dict:
    selected = set(selected_languages)
    mapped_languages = [
        language if language in selected else "Other"
        for language in record["source_languages"]
    ]
    out = dict(record)
    out["source_languages"] = sorted(set(mapped_languages))
    return out


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_labels(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

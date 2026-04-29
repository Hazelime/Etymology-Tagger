from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np

from .extract import read_jsonl
from .features import append_orthographic_features
from .model import load_model, predict_scores
from .vectors import load_vectors, vector_for_word

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+")

PALETTE = [
    "#2f80ed",
    "#219653",
    "#9b51e0",
    "#f2994a",
    "#eb5757",
    "#00a3a3",
    "#6f4e37",
    "#b83280",
    "#4f4f4f",
    "#1f7a8c",
]


class EtymologyPredictor:
    def __init__(self, config_path: str | Path = "configs/prototype.json"):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.labels = json.loads(Path(self.config["labels_path"]).read_text(encoding="utf-8"))
        self.model = load_model(Path(self.config["model_path"]))
        self.word_to_index, self.vectors = load_vectors(Path(self.config["vector_subset_path"]))
        self.records = {record["word"]: record for record in read_jsonl(Path(self.config["records_path"]))}
        self.language_colors = {
            label: PALETTE[idx % len(PALETTE)]
            for idx, label in enumerate(self.labels["source_languages"])
        }

    def predict_word(self, word: str) -> dict:
        clean = word.lower()
        x = vector_for_word(clean, self.word_to_index, self.vectors)[None, :]
        x = append_orthographic_features(x, [clean], self.config.get("orthographic_features"))
        lang_weights, lang_bias = self.model["language"]
        mech_weights, mech_bias = self.model["mechanism"]
        language_scores = predict_scores(x, lang_weights, lang_bias)[0]
        mechanism_scores = predict_scores(x, mech_weights, mech_bias)[0]
        lang_idx = int(np.argmax(language_scores))
        mech_idx = int(np.argmax(mechanism_scores))
        source_language = self.labels["source_languages"][lang_idx]
        mechanism = self.labels["mechanisms"][mech_idx]
        return {
            "word": word,
            "source_language": source_language,
            "source_language_score": float(language_scores[lang_idx]),
            "mechanism": mechanism,
            "mechanism_score": float(mechanism_scores[mech_idx]),
            "record": self.records.get(clean),
        }

    def annotate_html(self, text: str) -> str:
        spans = []
        for token in TOKEN_RE.findall(text):
            if token.isalpha() or ("'" in token and token.replace("'", "").isalpha()):
                prediction = self.predict_word(token)
                color = self.language_colors.get(prediction["source_language"], "#666666")
                detail = breakdown_text(prediction)
                spans.append(
                    "<button class='word-chip' "
                    f"style='--chip-color:{color}' "
                    f"onclick=\"showBreakdown({html.escape(json.dumps(detail), quote=True)})\">"
                    f"{html.escape(token)}"
                    "</button>"
                )
            else:
                spans.append(html.escape(token))
        return "".join(spans)


def paired_label(mechanism: str, language: str) -> str:
    if mechanism == "borrowed":
        return f"borrowed from {language}"
    if mechanism == "derived":
        return f"derived from {language}"
    if mechanism == "calqued":
        return f"calqued from {language}"
    if mechanism == "inherited":
        return f"inherited from {language}"
    return f"{mechanism} from {language}"


def breakdown_text(prediction: dict) -> str:
    record = prediction.get("record")
    lines = [
        f"{prediction['word']}",
        paired_label(prediction["mechanism"], prediction["source_language"]),
        (
            f"Model confidence: language {prediction['source_language_score']:.2f}, "
            f"mechanism {prediction['mechanism_score']:.2f}"
        ),
    ]
    if record:
        if record.get("pairs"):
            lines.append("")
            lines.append("Wiktionary-derived pairs:")
            for pair in record["pairs"][:12]:
                lines.append(f"- {paired_label(pair['mechanism'], pair['source_language'])}")
        if record.get("etymology_texts"):
            lines.append("")
            lines.append("Etymology:")
            lines.append(record["etymology_texts"][0][:1200])
    else:
        lines.append("")
        lines.append("No compact Wiktionary record was found for this word in the prototype dataset.")
    return "\n".join(lines)

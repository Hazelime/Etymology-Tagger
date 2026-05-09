from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import torch

from .extract import read_jsonl
from .features import append_orthographic_features
from .model import load_model
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

CLASSIFIER_OPTIONS = {
    "Y-Shape MTL Classifier": "mtl",
}


class EtymologyPredictor:
    def __init__(self, config_path: str | Path = "configs/prototype.json"):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.labels = json.loads(Path(self.config["labels_path"]).read_text(encoding="utf-8"))
        metadata_path = Path(self.config.get("metadata_path", "models/metadata.json"))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        
        input_dim = self.metadata.get("input_dim", 300) # fallback
        num_languages = len(self.labels["source_languages"])
        num_mechanisms = len(self.labels["mechanisms"])
        
        self.model = load_model(Path(self.config["model_path"]), input_dim, num_languages, num_mechanisms)
        self.word_to_index, self.vectors = load_vectors(Path(self.config["vector_subset_path"]))
        self.records = {record["word"]: record for record in read_jsonl(Path(self.config["records_path"]))}
        self.record_count = len(self.records)
        self.language_colors = {
            label: PALETTE[idx % len(PALETTE)]
            for idx, label in enumerate(self.labels["source_languages"])
        }
        all_counts = self.labels.get("language_counts", {})
        top_langs = set(self.labels["source_languages"])
        if "Other" in top_langs:
            top_langs.remove("Other")
            
        self.language_frequencies = {
            l: (all_counts.get(l, 0) / max(self.record_count, 1)) * 100
            for l in top_langs
        }
        
        other_count = sum(count for lang, count in all_counts.items() if lang not in top_langs)
        self.language_frequencies["Other"] = (other_count / max(self.record_count, 1)) * 100

    def predict_word(self, word: str, classifier_name: str = "Y-Shape MTL Classifier") -> dict:
        clean = word.lower()
        x = vector_for_word(clean, self.word_to_index, self.vectors)[None, :]
        x = append_orthographic_features(x, [clean], self.config.get("orthographic_features"))
        
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            lang_logits, mech_logits = self.model(x_t)
            language_scores = torch.sigmoid(lang_logits).numpy()[0]
            mechanism_scores = torch.sigmoid(mech_logits).numpy()[0]
        
        lang_idx = int(np.argmax(language_scores))
        mech_idx = int(np.argmax(mechanism_scores))
        
        source_language = self.labels["source_languages"][lang_idx]
        mechanism = self.labels["mechanisms"][mech_idx]
        
        language_predictions = self._selected_predictions(
            self.labels["source_languages"],
            language_scores,
            self.metadata.get("thresholds", {}).get("source_language"),
            lang_idx,
        )
        mechanism_predictions = self._selected_predictions(
            self.labels["mechanisms"],
            mechanism_scores,
            self.metadata.get("thresholds", {}).get("source_mechanism"),
            mech_idx,
        )
        return {
            "word": word,
            "source_language": source_language,
            "source_language_score": float(language_scores[lang_idx]),
            "mechanism": mechanism,
            "mechanism_score": float(mechanism_scores[mech_idx]),
            "source_language_predictions": language_predictions,
            "mechanism_predictions": mechanism_predictions,
            "classifier_name": classifier_name,
            "record": self.records.get(clean),
            "evaluation": self.metadata.get("evaluation"),
        }

    @staticmethod
    def _selected_predictions(
        labels: list[str], scores: np.ndarray, thresholds: list[float] | float | None, top_idx: int
    ) -> list[dict]:
        predictions = []
        for idx in np.argsort(-scores):
            score = float(scores[idx])
            t = thresholds[idx] if isinstance(thresholds, list) else (thresholds or 0.5)
            if idx == top_idx or score >= t:
                predictions.append({"label": labels[idx], "score": score})
            if len(predictions) >= 6:
                break
        return predictions

    def annotate_html(self, text: str) -> str:
        tokens = TOKEN_RE.findall(text)
        words_html = []
        panels_html = []
        
        for i, token in enumerate(tokens):
            if not re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)?$", token):
                words_html.append(f"<span>{html.escape(token)}</span>")
                continue
            
            prediction = self.predict_word(token)
            color = self.language_colors.get(prediction["source_language"], "#6b7280")
            panel_id = f"panel_{i}"
            
            words_html.append(
                f"<span class='etym-word' tabindex='0' style='--language-color: {color};' "
                f"onclick='showPanel(\"{panel_id}\")'>{html.escape(token)}</span>"
            )
            
            panels_html.append(
                f"<div id='{panel_id}' class='breakdown-panel'>"
                f"{breakdown_text(prediction)}"
                f"</div>"
            )
            
        return (
            "<div class='etag-container'>"
            + "<div class='tagged-output'>"
            + "".join(words_html)
            + "</div>"
            + "<div class='breakdown-stack'>"
            + "<div class='breakdown-placeholder'>Click on a word to view its etymological breakdown.</div>"
            + "".join(panels_html)
            + "</div>"
            + "</div>"
        )


def _format_predictions(predictions: list[dict], max_items: int = 6) -> str:
    visible = predictions[:max_items]
    text = ", ".join(f"{item['label']} ({item['score']:.2f})" for item in visible)
    if len(predictions) > max_items:
        text += f", +{len(predictions) - max_items} more"
    return text


def _clean_etymology_text(text: str) -> str:
    text = text.strip()
    if not text.startswith("Etymology tree"):
        return text
    markers = [
        "\nFrom ",
        "\nBorrowed ",
        "\nInherited ",
        "\nLearned borrowing ",
        "\nUltimately ",
        "\nPossibly ",
        "\nProbably ",
    ]
    positions = [text.find(marker) for marker in markers if text.find(marker) != -1]
    if positions:
        return text[min(positions) + 1 :].strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-8:]).strip()


def breakdown_text(prediction: dict) -> str:
    record = prediction.get("record")
    lines = [
        f"<b>{html.escape(prediction['word'])}</b>",
        "<br>",
        f"Predicted source language: <b>{html.escape(prediction['source_language'])}</b> ({prediction['source_language_score']:.2f})",
        "<br>",
        f"Predicted entry mechanism: <b>{html.escape(prediction['mechanism'])}</b> ({prediction['mechanism_score']:.2f})",
        "<br><hr>",
        "<b>Top source language candidates:</b><br>",
        f"{html.escape(_format_predictions(prediction['source_language_predictions']))}",
        "<br><br>",
        "<b>Top entry mechanism candidates:</b><br>",
        f"{html.escape(_format_predictions(prediction['mechanism_predictions']))}",
        "<hr>",
    ]
    if record:
        if record.get("pairs"):
            lines.append("<b>Wiktionary gold standard:</b><br>")
            seen_pairs = set()
            for pair in record["pairs"][:12]:
                pair_str = f"{pair['mechanism']} from {pair['source_language']}"
                if pair_str not in seen_pairs:
                    lines.append(f"- {html.escape(pair_str)}<br>")
                    seen_pairs.add(pair_str)
        if record.get("etymology_texts"):
            cleaned = _clean_etymology_text(record["etymology_texts"][0])
            lines.append("<br>")
            lines.append("<b>Wiktionary etymological notes:</b><br>")
            lines.append(html.escape(cleaned[:1200]))
    else:
        lines.append("<br>")
        lines.append("No compact Wiktionary record was found for this word in the training dataset.")
            
    return "".join(lines)

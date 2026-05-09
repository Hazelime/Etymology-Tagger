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

# Regular expression to tokenize text into words (including apostrophes) and non-word separators.
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^A-Za-z]+")

# Color palette for highlighting different source languages in the UI.
PALETTE = [
    "#2f80ed", # Blue (Latinate)
    "#219653", # Green (Germanic)
    "#9b51e0", # Purple (Greek)
    "#f2994a", # Orange (Romance)
    "#eb5757", # Red (Other)
    "#00a3a3",
    "#6f4e37",
    "#b83280",
    "#4f4f4f",
    "#1f7a8c",
]

class EtymologyPredictor:
    """
    Main inference class for the Etymology Tagger.
    
    Handles loading the trained PyTorch model, word vector subsets, and providing 
    predictions with formatted HTML output for the Gradio frontend.
    """
    def __init__(self, config_path: str | Path = "configs/prototype.json"):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.labels = json.loads(Path(self.config["labels_path"]).read_text(encoding="utf-8"))
        metadata_path = Path(self.config.get("metadata_path", "models/metadata.json"))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        
        input_dim = self.metadata.get("input_dim", 300)
        num_languages = len(self.labels["source_languages"])
        num_mechanisms = len(self.labels["mechanisms"])
        
        # Load the Y-Shape MTL model weights
        self.model = load_model(Path(self.config["model_path"]), input_dim, num_languages, num_mechanisms)
        
        # Load the word vector subset (fastText)
        self.word_to_index, self.vectors = load_vectors(Path(self.config["vector_subset_path"]))
        
        # Load ground truth records for verification/fallback in the UI
        self.records = {record["word"]: record for record in read_jsonl(Path(self.config["records_path"]))}
        self.record_count = len(self.records)
        
        # Precompute color mappings and background frequencies for the UI legend
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

    def predict_word(self, word: str) -> dict:
        """
        Runs inference on a single word.
        
        Returns a dictionary containing the top predictions for both language and mechanism,
        along with the confidence scores.
        """
        clean = word.lower()
        
        # 1. Featurize: Prepend semantic vector + hashed orthographic features
        x = vector_for_word(clean, self.word_to_index, self.vectors)[None, :]
        x = append_orthographic_features(x, [clean], self.config.get("orthographic_features"))
        
        # 2. PyTorch Inference
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            lang_logits, mech_logits = self.model(x_t)
            # Apply sigmoid to get probabilities (since it's a multi-label setup)
            language_scores = torch.sigmoid(lang_logits).numpy()[0]
            mechanism_scores = torch.sigmoid(mech_logits).numpy()[0]
        
        # 3. Select top candidates
        lang_idx = int(np.argmax(language_scores))
        mech_idx = int(np.argmax(mechanism_scores))
        
        source_language = self.labels["source_languages"][lang_idx]
        mechanism = self.labels["mechanisms"][mech_idx]
        
        # Apply tuned thresholds to filter out low-confidence multi-label candidates
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
            "record": self.records.get(clean),
            "evaluation": self.metadata.get("evaluation"),
        }

    @staticmethod
    def _selected_predictions(
        labels: list[str], scores: np.ndarray, thresholds: list[float] | float | None, top_idx: int
    ) -> list[dict]:
        """Filters model scores using per-class thresholds."""
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
        """
        Annotates raw text with interactive HTML/CSS highlighting.
        
        Each word becomes clickable, revealing a detailed etymological breakdown 
        in a side panel.
        """
        tokens = TOKEN_RE.findall(text)
        words_html = []
        panels_html = []
        
        for i, token in enumerate(tokens):
            # Only process word tokens
            if not re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)?$", token):
                words_html.append(f"<span>{html.escape(token)}</span>")
                continue
            
            prediction = self.predict_word(token)
            color = self.language_colors.get(prediction["source_language"], "#6b7280")
            panel_id = f"panel_{i}"
            
            # Create the interactive word span
            words_html.append(
                f"<span class='etym-word' tabindex='0' style='--language-color: {color};' "
                f"onclick='showPanel(\"{panel_id}\")'>{html.escape(token)}</span>"
            )
            
            # Create the hidden breakdown panel for this word
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
    """Helper for formatting score lists in the UI."""
    visible = predictions[:max_items]
    text = ", ".join(f"{item['label']} ({item['score']:.2f})" for item in visible)
    if len(predictions) > max_items:
        text += f", +{len(predictions) - max_items} more"
    return text

def _clean_etymology_text(text: str) -> str:
    """Strips verbose Wiktextract headers from ground-truth text."""
    text = text.strip()
    if not text.startswith("Etymology tree"):
        return text
    markers = ["\nFrom ", "\nBorrowed ", "\nInherited ", "\nLearned borrowing ", "\nUltimately "]
    positions = [text.find(marker) for marker in markers if text.find(marker) != -1]
    if positions:
        return text[min(positions) + 1 :].strip()
    return "\n".join(text.splitlines()[-8:]).strip()

def breakdown_text(prediction: dict) -> str:
    """Generates the content for the word breakdown panel."""
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
            lines.append("<br><b>Wiktionary etymology text:</b><br>")
            text = _clean_etymology_text(record["etymology_texts"][0])
            lines.append(f"<div class='etym-raw'>{html.escape(text)}</div>")
            
    return "".join(lines)

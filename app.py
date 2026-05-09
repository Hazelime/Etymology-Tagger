from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from etymology_tagger.predict import CLASSIFIER_OPTIONS, EtymologyPredictor

CSS = """
.legend {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  column-gap: 24px;
  row-gap: 8px;
  margin: 8px 0 14px;
  align-items: center;
}
.legend-item {
  display: grid;
  grid-template-columns: 14px 1fr;
  align-items: center;
  column-gap: 8px;
  font-size: 13px;
  line-height: 1.25;
  color: #111827;
}
.legend-note {
  margin: -4px 0 12px;
  color: #ffffff;
  font-size: 12px;
}
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.etag-radio {
  display: none !important;
}
.tagged-output {
  line-height: 1.8;
  font-size: 16px;
  min-height: 132px;
  border: 1px solid #d5d8de;
  border-radius: 6px;
  padding: 12px;
  white-space: pre-wrap;
}
.etym-word {
  display: inline !important;
  border-bottom: 2px solid color-mix(in srgb, var(--language-color) 42%, transparent);
  border-radius: 3px;
  cursor: pointer;
  padding: 0 1px;
  transition: color 120ms ease, background-color 120ms ease, font-weight 120ms ease;
}
.etym-word:hover,
.etym-word:focus {
  color: var(--language-color) !important;
  background: color-mix(in srgb, var(--language-color) 12%, transparent);
  font-weight: 700;
  outline: none;
}
.breakdown-stack {
  margin-top: 12px;
}
.breakdown-panel {
  display: none;
  min-height: 160px;
  white-space: pre-wrap;
  border: 1px solid #d5d8de;
  border-radius: 6px;
  padding: 12px;
  line-height: 1.45;
  font-size: 14px;
  text-align: left;
}
.breakdown-placeholder {
  min-height: 80px;
  border: 1px dashed #d5d8de;
  border-radius: 6px;
  padding: 12px;
  font-size: 14px;
}
.eval-section {
  margin-top: 32px;
  padding-top: 24px;
  border-top: 1px solid #374151;
}
.eval-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  color: #ffffff;
  background: #000000;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}
.eval-table th {
  background: #1f2937;
  font-weight: 600;
  text-align: left;
  padding: 10px 16px;
  border-bottom: 1px solid #374151;
  color: #ffffff;
}
.eval-table td {
  padding: 10px 16px;
  border-bottom: 1px solid #111827;
  color: #ffffff;
}
.eval-table tr:last-child td {
  border-bottom: none;
}
.eval-title {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #ffffff;
}
"""

predictor = EtymologyPredictor()


def legend_html() -> str:
    items = []
    for language, color in predictor.language_colors.items():
        frequency = predictor.language_frequencies.get(language, 0.0)
        items.append(
            f"<span class='legend-item'><span class='swatch' style='background:{color}'></span>"
            f"<span>{language} ({frequency:.2f}%)</span></span>"
        )
    return (
        "<div class='legend'>"
        + "".join(items)
        + "</div><div class='legend-note'>Labels can overlap. Percentages are based on word types in the training vocabulary.</div>"
    )


def evaluation_html() -> str:
    eval_data = predictor.metadata.get("evaluation", {})
    if not eval_data:
        return ""
    
    rows = []
    for head in ["source_language", "source_mechanism"]:
        m = eval_data.get(head, {})
        name = "Source Language" if "language" in head else "Entry Mechanism"
        rows.append(
            f"<tr>"
            f"<td>{name}</td>"
            f"<td>{m.get('precision', 0):.4f}</td>"
            f"<td>{m.get('recall', 0):.4f}</td>"
            f"<td>{m.get('f1', 0):.4f}</td>"
            f"</tr>"
        )
    
    return (
        "<div class='eval-section'>"
        "<div class='eval-title'>Model Performance (Held-out Test Set)</div>"
        "<table class='eval-table'>"
        "<thead><tr><th>Component</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
        "</div>"
    )


def tag_text(text: str) -> str:
    if not text.strip():
        return "<div class='etag-result'><div class='tagged-output'></div></div>"
    return (
        "<div class='etag-result'>"
        + legend_html()
        + predictor.annotate_html(text)
        + "</div>"
    )


JS = """
function showPanel(id) {
    document.querySelectorAll('.breakdown-panel').forEach(p => p.style.display = 'none');
    const placeholder = document.querySelector('.breakdown-placeholder');
    if(placeholder) placeholder.style.display = 'none';
    const panel = document.getElementById(id);
    if(panel) panel.style.display = 'block';
}
"""

with gr.Blocks(css=CSS, js=JS) as demo:
    gr.Markdown("# English Etymology Tagger")
    text = gr.Textbox(
        label="Text to tag",
        lines=5,
        value="The berserk corgi said 'tycoon' from the jungle as the cosmonaut sought chaos with an avocado.",
    )
    button = gr.Button("Tag text", variant="primary")
    output = gr.HTML(label="Tagged text")
    gr.HTML(evaluation_html())
    button.click(tag_text, inputs=[text], outputs=output)
    text.submit(tag_text, inputs=[text], outputs=output)
    demo.load(tag_text, inputs=[text], outputs=output)


if __name__ == "__main__":
    demo.launch()

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from etymology_tagger.predict import EtymologyPredictor

CSS = """
.legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 14px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.tagged-output { line-height: 2.25; font-size: 18px; }
.word-chip {
  border: 0;
  border-bottom: 3px solid var(--chip-color);
  background: color-mix(in srgb, var(--chip-color) 16%, white);
  color: #1f2933;
  padding: 2px 5px;
  margin: 0 1px;
  border-radius: 4px;
  cursor: pointer;
  font: inherit;
}
.word-chip:hover { background: color-mix(in srgb, var(--chip-color) 26%, white); }
#breakdown {
  min-height: 180px;
  white-space: pre-wrap;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 12px;
  background: #fafafa;
}
"""

SCRIPT = """
<script>
function showBreakdown(text) {
  const box = document.getElementById("breakdown");
  if (box) box.textContent = text;
}
</script>
"""

predictor = EtymologyPredictor()


def legend_html() -> str:
    items = []
    for language, color in predictor.language_colors.items():
        items.append(
            f"<span class='legend-item'><span class='swatch' style='background:{color}'></span>{language}</span>"
        )
    return "<div class='legend'>" + "".join(items) + "</div>"


def tag_text(text: str) -> str:
    if not text.strip():
        return "<div class='tagged-output'></div>" + SCRIPT
    return (
        legend_html()
        + "<div class='tagged-output'>"
        + predictor.annotate_html(text)
        + "</div>"
        + SCRIPT
    )


with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# English Etymology Tagger")
    text = gr.Textbox(
        label="Text to tag",
        lines=5,
        value="Dictionary, free, thesaurus, encyclopedia, cat, book, and elephant show different etymological paths.",
    )
    button = gr.Button("Tag text", variant="primary")
    output = gr.HTML(label="Tagged text")
    gr.HTML("<div id='breakdown'>Click a tagged word to see its etymological breakdown.</div>")
    button.click(tag_text, inputs=text, outputs=output)
    text.submit(tag_text, inputs=text, outputs=output)
    demo.load(tag_text, inputs=text, outputs=output)


if __name__ == "__main__":
    demo.launch()

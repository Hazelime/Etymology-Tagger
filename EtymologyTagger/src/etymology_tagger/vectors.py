from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .storage import assert_under_budget


def download_file(url: str, path: Path, project_root: Path, storage_budget_gb: float) -> None:
    request = Request(url, headers={"User-Agent": "EtymologyTagger/0.1"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=60) as response:
        content_length = int(response.headers.get("Content-Length") or 0)
        if content_length:
            assert_under_budget(project_root, storage_budget_gb, content_length)
        with path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def extract_vector_subset_from_zip(zip_path: Path, output_path: Path, vocab: set[str]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    with zipfile.ZipFile(zip_path) as archive:
        vec_members = [name for name in archive.namelist() if name.endswith(".vec")]
        if not vec_members:
            raise RuntimeError(f"No .vec member found in {zip_path}")
        with archive.open(vec_members[0]) as raw, output_path.open("w", encoding="utf-8") as out:
            header = raw.readline().decode("utf-8", errors="ignore").strip().split()
            dim = int(header[1]) if len(header) == 2 and header[1].isdigit() else None
            buffered_rows = []
            if dim is None:
                line = " ".join(header)
                word = line.split(" ", 1)[0]
                if word in vocab:
                    buffered_rows.append(line)
            for raw_line in raw:
                line = raw_line.decode("utf-8", errors="ignore").rstrip("\n")
                if not line:
                    continue
                word = line.split(" ", 1)[0]
                if word in vocab:
                    buffered_rows.append(line)
                    found += 1
                if found >= len(vocab):
                    break
            if dim is None and buffered_rows:
                dim = len(buffered_rows[0].split()) - 1
            if dim is None:
                dim = 300
            out.write(f"{len(buffered_rows)} {dim}\n")
            for row in buffered_rows:
                out.write(row + "\n")
    return found


def load_vectors(path: Path) -> tuple[dict[str, int], np.ndarray]:
    words: dict[str, int] = {}
    vectors = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip().split()
        has_header = len(first) == 2 and first[0].isdigit() and first[1].isdigit()
        if not has_header and first:
            word, values = first[0], first[1:]
            words[word] = 0
            vectors.append([float(v) for v in values])
        for line in handle:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            word, values = parts[0], parts[1:]
            try:
                vector = [float(v) for v in values]
            except ValueError:
                continue
            words[word] = len(vectors)
            vectors.append(vector)
    if not vectors:
        raise RuntimeError(f"No vectors loaded from {path}")
    matrix = np.asarray(vectors, dtype=np.float32)
    return words, matrix


def vector_for_word(word: str, word_to_index: dict[str, int], vectors: np.ndarray) -> np.ndarray:
    idx = word_to_index.get(word.lower())
    if idx is not None:
        return vectors[idx]
    return np.zeros(vectors.shape[1], dtype=np.float32)


def write_tiny_vectors(path: Path, words: list[str], dim: int = 32, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(words)} {dim}\n")
        for word in words:
            vector = rng.normal(0.0, 0.2, size=dim)
            values = " ".join(f"{x:.6f}" for x in vector)
            handle.write(f"{word} {values}\n")

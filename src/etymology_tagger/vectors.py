from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .storage import assert_under_budget

def download_file(url: str, path: Path, project_root: Path, storage_budget_gb: float) -> None:
    """
    Downloads a file from a URL to a local path while monitoring storage budgets.
    
    This is used to fetch the large fastText archive (approx. 4GB) before 
    subsetting it.
    """
    request = Request(url, headers={"User-Agent": "EtymologyTagger/0.1"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=60) as response:
        content_length = int(response.headers.get("Content-Length") or 0)
        if content_length:
            # Safety check: Ensure we don't exceed the user's disk space budget
            assert_under_budget(project_root, storage_budget_gb, content_length)
        with path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024) # 1MB chunks
                if not chunk:
                    break
                handle.write(chunk)

def extract_vector_subset_from_zip(zip_path: Path, output_path: Path, vocab: set[str]) -> int:
    """
    Parses a compressed fastText .zip archive and extracts vectors for a specific vocabulary.
    
    Motivation: Official fastText models contain millions of vectors. For our 
    specialized etymology task, we only need vectors for the ~100k words found 
    in our Wiktionary subset. This reduces the model size from ~4GB to ~80MB.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    with zipfile.ZipFile(zip_path) as archive:
        vec_members = [name for name in archive.namelist() if name.endswith(".vec")]
        if not vec_members:
            raise RuntimeError(f"No .vec member found in {zip_path}")
            
        with archive.open(vec_members[0]) as raw, output_path.open("w", encoding="utf-8") as out:
            header = raw.readline().decode("utf-8", errors="ignore").strip().split()
            # fastText .vec files start with a header: [num_words] [dim]
            dim = int(header[1]) if len(header) == 2 and header[1].isdigit() else None
            buffered_rows = []
            
            if dim is None:
                # Handle cases without a proper header line
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
                    # Optimization: Stop early if we found every word in our vocab
                    break
                    
            if dim is None and buffered_rows:
                dim = len(buffered_rows[0].split()) - 1
            if dim is None:
                dim = 300
                
            # Write a new, compact .vec file with a fresh header
            out.write(f"{len(buffered_rows)} {dim}\n")
            for row in buffered_rows:
                out.write(row + "\n")
                
    return found

def load_vectors(path: Path) -> tuple[dict[str, int], np.ndarray]:
    """
    Loads pretrained word vectors from a .vec file into a NumPy matrix.
    
    Returns:
        - A dictionary mapping words to their row index in the matrix.
        - A NumPy matrix of shape (num_words, dimensions).
    """
    words: dict[str, int] = {}
    vectors = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip().split()
        # Handle files with or without the fastText-style header
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
    """
    Retrieves the vector for a word, with fallback logic for multi-word or hyphenated terms.
    
    Etymological entries often include phrases (e.g., 'hot dog'). If the full phrase 
    isn't in fastText, we average the vectors of the individual components.
    """
    word = word.lower().strip()
    idx = word_to_index.get(word)
    if idx is not None:
        return vectors[idx]
    
    # Fallback: Try splitting on spaces and hyphens
    parts = [p for p in re.split(r"[ -]", word) if p]
    if len(parts) > 1:
        found_vectors = []
        for p in parts:
            p_idx = word_to_index.get(p)
            if p_idx is not None:
                found_vectors.append(vectors[p_idx])
        if found_vectors:
            # Return the centroid of the component vectors
            return np.mean(found_vectors, axis=0)
            
    # Final fallback: Return zero vector (the model will rely on orthographic features instead)
    return np.zeros(vectors.shape[1], dtype=np.float32)

from __future__ import annotations

import hashlib

import numpy as np


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def orthographic_feature_names(word: str, config: dict) -> list[str]:
    word = word.lower()
    if not word:
        return []
    min_ngram = int(config.get("min_char_ngram", 3))
    max_ngram = int(config.get("max_char_ngram", 5))
    max_prefix = int(config.get("max_prefix", 5))
    max_suffix = int(config.get("max_suffix", 5))
    padded = f"<{word}>"
    names = [f"len:{min(len(word), 20)}"]
    for n in range(1, max_prefix + 1):
        if len(word) >= n:
            names.append(f"pre{n}:{word[:n]}")
    for n in range(1, max_suffix + 1):
        if len(word) >= n:
            names.append(f"suf{n}:{word[-n:]}")
    for n in range(min_ngram, max_ngram + 1):
        if len(padded) >= n:
            for start in range(0, len(padded) - n + 1):
                names.append(f"chr{n}:{padded[start:start+n]}")
    return names


def orthographic_vector(word: str, config: dict) -> np.ndarray:
    dim = int(config.get("hash_dim", 512))
    vector = np.zeros(dim, dtype=np.float32)
    names = orthographic_feature_names(word, config)
    if not names:
        return vector
    scale = 1.0 / np.sqrt(len(names))
    for name in names:
        hashed = stable_hash(name)
        index = hashed % dim
        sign = 1.0 if ((hashed >> 63) & 1) == 0 else -1.0
        vector[index] += sign * scale
    return vector


def append_orthographic_features(x: np.ndarray, words: list[str], config: dict | None) -> np.ndarray:
    if not config or not config.get("enabled", False):
        return x
    ortho = np.vstack([orthographic_vector(word, config) for word in words])
    return np.hstack([x, ortho]).astype(np.float32)

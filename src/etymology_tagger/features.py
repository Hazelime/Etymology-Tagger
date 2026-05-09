from __future__ import annotations

import hashlib
import numpy as np

def stable_hash(text: str) -> int:
    """
    Computes a stable 64-bit hash for a string.
    
    We use Blake2b instead of the built-in hash() because Python's hash() is 
    randomized per-session for security, which would break model consistency 
    between training and inference.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)

def orthographic_feature_names(word: str, config: dict) -> list[str]:
    """
    Extracts high-level orthographic signals from a word.
    
    Includes:
    - Prefixes/Suffixes (e.g., 'pre1:t', 'suf2:on')
    - Character N-grams (e.g., 'chr3:<th', 'chr3:the', 'chr3:he>')
    - Word length
    
    These patterns are crucial for etymology because they capture morphological 
    markers (like the Latinate '-ation') that fastText vectors might miss.
    """
    word = word.lower()
    if not word:
        return []
    min_ngram = int(config.get("min_char_ngram", 3))
    max_ngram = int(config.get("max_char_ngram", 5))
    max_prefix = int(config.get("max_prefix", 5))
    max_suffix = int(config.get("max_suffix", 5))
    
    # Pad word with markers to detect start/end patterns specifically
    padded = f"<{word}>"
    names = [f"len:{min(len(word), 20)}"]
    
    # 1. Extract prefixes
    for n in range(1, max_prefix + 1):
        if len(word) >= n:
            names.append(f"pre{n}:{word[:n]}")
            
    # 2. Extract suffixes
    for n in range(1, max_suffix + 1):
        if len(word) >= n:
            names.append(f"suf{n}:{word[-n:]}")
            
    # 3. Extract character n-grams
    for n in range(min_ngram, max_ngram + 1):
        if len(padded) >= n:
            for start in range(0, len(padded) - n + 1):
                names.append(f"chr{n}:{padded[start:start+n]}")
                
    return names

def orthographic_vector(word: str, config: dict) -> np.ndarray:
    """
    Maps orthographic feature names to a fixed-size vector using Feature Hashing.
    
    Feature Hashing (the 'Hashing Trick') allows us to handle an open-ended number 
    of character patterns without needing a massive vocabulary dictionary.
    
    We use 'Signed Hashing' to reduce the impact of hash collisions.
    """
    dim = int(config.get("hash_dim", 512))
    vector = np.zeros(dim, dtype=np.float32)
    names = orthographic_feature_names(word, config)
    if not names:
        return vector
        
    # Scale the vector by the number of features to prevent long words 
    # from having excessively large activations.
    scale = 1.0 / np.sqrt(len(names))
    
    for name in names:
        hashed = stable_hash(name)
        index = hashed % dim
        # Determine sign bit for signed hashing
        sign = 1.0 if ((hashed >> 63) & 1) == 0 else -1.0
        vector[index] += sign * scale
        
    return vector

def append_orthographic_features(x: np.ndarray, words: list[str], config: dict | None) -> np.ndarray:
    """
    Appends the hashed orthographic vector to the existing semantic vector.
    
    This creates a hybrid representation that combines semantic context 
    (from fastText) with morphological cues.
    """
    if not config or not config.get("enabled", False):
        return x
    ortho = np.vstack([orthographic_vector(word, config) for word in words])
    return np.hstack([x, ortho]).astype(np.float32)

"""Default embedder: hashed 1-3-gram token bag, L2-normalised, no network (D4.4.3)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9_.]+")


class HashingEmbedder:
    """Deterministic ``contracts.Embedder``: token 1-3-grams hashed into ``dim`` buckets."""

    def __init__(self, dim: int = 512, max_n: int = 3) -> None:
        self.dim = dim
        self.max_n = max_n

    def embed(self, text: str) -> tuple[float, ...]:
        toks = _TOKEN.findall(text.lower())
        vec = [0.0] * self.dim
        for n in range(1, self.max_n + 1):
            for i in range(len(toks) - n + 1):
                h = hashlib.blake2b(" ".join(toks[i:i + n]).encode(), digest_size=8).digest()
                x = int.from_bytes(h, "big")
                vec[x % self.dim] += 1.0 if (x >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return tuple(vec)
        return tuple(v / norm for v in vec)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 for a zero vector or mismatched dimensions."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def changed_lines(diff_text: str) -> str:
    """Only the +/- lines of a unified diff (context lines make unrelated diffs look alike)."""
    keep = [ln[1:] for ln in diff_text.splitlines()
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    return "\n".join(keep)

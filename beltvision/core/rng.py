"""Deterministic seeded randomness.

A pipeline run is a pure function of ``(params, seed)``. Every source of randomness
in the pipeline must draw from a generator produced here so that a run is
reproducible and a committed artifact can be regenerated bit-for-bit.
"""
from __future__ import annotations

import hashlib

import numpy as np

# The default seed nods to the reference case (CCTV camera COLA 34).
DEFAULT_SEED: int = 34


def seeded_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a NumPy Generator seeded deterministically.

    Use one generator per case run and thread it through the stages rather than
    calling the global ``numpy.random`` functions, which are not reproducible.
    """
    return np.random.default_rng(int(seed))


def derive_seed(seed: int, salt: str) -> int:
    """Derive a stable child seed from a parent seed and a string salt.

    Lets a stage take an independent, still-deterministic sub-stream (for example
    per-case or per-method) without disturbing the parent stream.
    """
    digest = hashlib.sha256(f"{int(seed)}:{salt}".encode()).digest()[:8]
    value = int.from_bytes(digest, "big")
    return value % (2**31 - 1)

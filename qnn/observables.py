from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np


def z_chain_expectation_statevector(probs: np.ndarray) -> float:
    """Compute <Z⊗...⊗Z> from a probability vector (length 2^q).
    Assumes index i uses little-endian bit order with qubit 0 as LSB.
    """
    exp = 0.0
    for i, p in enumerate(probs):
        parity = bin(i).count("1") & 1
        exp += float(p) * (-1.0 if parity else 1.0)
    return exp


def sample_from_probs(probs: np.ndarray, shots: int) -> List[int]:
    idxs = np.arange(len(probs))
    draws = np.random.choice(idxs, size=shots, p=probs)
    return draws.tolist()


def z_chain_expectation_from_samples(samples: List[int]) -> float:
    exp = 0.0
    for i in samples:
        parity = bin(i).count("1") & 1
        exp += (-1.0 if parity else 1.0)
    return exp / max(1, len(samples))


def index_to_bitstring(i: int, q: int) -> str:
    return format(i, f"0{q}b")[::-1]  # little-endian qubit order


__all__ = [
    "z_chain_expectation_statevector",
    "sample_from_probs",
    "z_chain_expectation_from_samples",
    "index_to_bitstring",
]

def zz_grid_expectation_statevector(probs: np.ndarray, H: int, W: int) -> float:
    """Compute average ZZ over 4-neighborhood edges on an HxW grid mapping (row-major) for first H*W qubits.
    Assumes little-endian bit order. Returns mean over all considered edges.
    """
    q = H * W
    edges = []
    def idx(h, w):
        return h * W + w
    for h in range(H):
        for w in range(W):
            a = idx(h, w)
            if w + 1 < W:
                edges.append((a, idx(h, w + 1)))
            if h + 1 < H:
                edges.append((a, idx(h + 1, w)))
    if not edges:
        return 0.0
    # Expectation of Z_i Z_j = sum_s p(s) z_i(s) z_j(s) with z_k(s) in {+1,-1}
    exp_sum = 0.0
    for i, p in enumerate(probs):
        # bitstring little-endian
        for (a, b) in edges:
            za = -1.0 if (i >> a) & 1 else 1.0
            zb = -1.0 if (i >> b) & 1 else 1.0
            exp_sum += float(p) * (za * zb)
    return exp_sum / len(edges)

__all__.append("zz_grid_expectation_statevector")

def z_expectation_statevector(probs: np.ndarray, qubit: int) -> float:
    """Compute <Z_qubit> from probability vector assuming little-endian order."""
    exp = 0.0
    for i, p in enumerate(probs):
        z = -1.0 if (i >> qubit) & 1 else 1.0
        exp += float(p) * z
    return exp

__all__.append("z_expectation_statevector")

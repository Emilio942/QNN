from __future__ import annotations

from typing import Dict

from qiskit import transpile


def summarize_circuit(qc) -> Dict:
    ops = qc.count_ops()
    twoq = sum(v for k, v in ops.items() if k.lower() in {"cx", "cz", "swap", "iswap"})
    return {
        "qubits": qc.num_qubits,
        "depth": qc.depth(),
        "size": qc.size(),
        "ops": {str(k): int(v) for k, v in ops.items()},
        "two_qubit_ops": int(twoq),
    }


def transpile_for_backend(qc, backend, optimization_level: int = 1):
    tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
    return tqc


__all__ = ["summarize_circuit", "transpile_for_backend"]

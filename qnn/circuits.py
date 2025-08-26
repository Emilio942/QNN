from __future__ import annotations

from typing import List, Dict, Any, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation


def angle_encode_circuit(angles: List[float], q: int, L: int, entangler: str = "cz-ring", entangler_config: Optional[Dict[str, int]] = None) -> QuantumCircuit:
    """Build a simple angle-encoding circuit: L layers, each loads q angles into Ry then entangles.
    angles: length should be >= q*L; extra values ignored.
    """
    qc = QuantumCircuit(q)
    idx = 0
    for _ in range(L):
        # load angles
        for j in range(q):
            if idx < len(angles):
                qc.ry(angles[idx], j)
            idx += 1
        # entangle
        if entangler == "cz-ring":
            for j in range(q):
                qc.cz(j, (j + 1) % q)
        elif entangler == "cx-chain":
            for j in range(q - 1):
                qc.cx(j, j + 1)
        elif entangler == "cz-grid":
            # Expect entangler_config with H and W; map first q qubits row-major on HxW
            if not entangler_config or "H" not in entangler_config or "W" not in entangler_config:
                raise ValueError("cz-grid requires entangler_config with H and W")
            H = int(entangler_config["H"]) ; W = int(entangler_config["W"]) ;
            # Sanity: q should be <= H*W
            # Add CZ between right and down neighbors within the first q positions
            def idx(h, w):
                return h * W + w
            for h in range(H):
                for w in range(W):
                    a = idx(h, w)
                    if a >= q:
                        continue
                    if w + 1 < W:
                        b = idx(h, w + 1)
                        if b < q:
                            qc.cz(a, b)
                    if h + 1 < H:
                        b = idx(h + 1, w)
                        if b < q:
                            qc.cz(a, b)
        else:
            pass
    return qc


__all__ = ["angle_encode_circuit"]
 
def amplitude_prep_circuit(amplitudes: List[float]) -> QuantumCircuit:
    """Build a circuit that prepares |psi> = sum_i amplitudes[i] |i>.
    Length must be a power of two and L2-normalized. Uses Qiskit's StatePreparation.
    """
    import math
    n = int(math.log2(len(amplitudes)))
    if 2 ** n != len(amplitudes):
        raise ValueError("Amplitude vector length must be a power of two")
    # Qiskit expects a statevector (complex); accept real amplitudes
    sp = StatePreparation(amplitudes)
    qc = QuantumCircuit(n)
    qc.append(sp, range(n))
    return qc

__all__.append("amplitude_prep_circuit")

def phase_encode_circuit(phases: List[float], q: int, L: int, entangler: str = "cz-ring") -> QuantumCircuit:
    """Build a simple phase-encoding circuit: initialize |0>^q, optional H layer to create superposition,
    then apply Rz with provided phases in L re-upload layers and entangle.
    """
    qc = QuantumCircuit(q)
    # Optional layer to spread amplitudes (use H gates)
    for j in range(q):
        qc.h(j)
    idx = 0
    for _ in range(L):
        for j in range(q):
            if idx < len(phases):
                qc.rz(phases[idx], j)
            idx += 1
        if entangler == "cz-ring":
            for j in range(q):
                qc.cz(j, (j + 1) % q)
        elif entangler == "cx-chain":
            for j in range(q - 1):
                qc.cx(j, j + 1)
        else:
            pass
    return qc

__all__.append("phase_encode_circuit")

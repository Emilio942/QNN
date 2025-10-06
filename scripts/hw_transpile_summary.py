#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.preprocess import preprocess_vector
from qnn.planning import plan_from_spec
from qnn.vqa import build_reupload_variational_circuit
from qnn.transpile import summarize_circuit

from qiskit.transpiler import CouplingMap


def main():
    parser = argparse.ArgumentParser(description="Hardware-targeted transpile summary")
    parser.add_argument("--spec", type=str, default=str(ROOT / "specs" / "tensor_spec.yaml"))
    parser.add_argument("--q", type=int, default=4, help="Data qubits")
    parser.add_argument("--L", type=int, default=2, help="Re-upload layers")
    parser.add_argument("--basis", type=str, nargs="*", default=["rz", "sx", "x", "cx"], help="Basis gates")
    parser.add_argument("--coupling", type=str, help="Coupling map as JSON string e.g. '[[0,1],[1,2]]'")
    parser.add_argument("--preset", type=str, help="Preset topology, e.g. linear3, ring3, heavy-hex")
    parser.add_argument("--out", type=str, default=str(ROOT / "reports" / "hw_transpile_summary.json"))
    args = parser.parse_args()

    spec = load_tensor_spec(args.spec)
    d = spec["shape"]["d"] if spec["shape"]["layout"] == "d" else spec["shape"]["H"] * spec["shape"]["W"] * spec["shape"]["C"]
    vec = [0.0] * int(d)
    v, info = preprocess_vector(vec, spec)
    angles = info.get("angles") or [0.0] * args.q
    qc = build_reupload_variational_circuit(args.q, args.L, angles, [0.0] * (args.q * args.L))

    # Handle presets (may set default basis/coupling; explicit flags override)
    basis = list(args.basis) if args.basis else ["rz", "sx", "x", "cx"]
    coupling_edges = None
    if args.preset:
        preset = args.preset.strip().lower()
        m = re.match(r"^(linear|ring)(\d+)?$", preset)
        if m:
            kind, n_str = m.group(1), m.group(2)
            n = int(n_str) if n_str else int(args.q)
            if kind == "linear":
                coupling_edges = [[i, i + 1] for i in range(max(0, n - 1))]
            else:  # ring
                coupling_edges = [[i, (i + 1) % n] for i in range(max(0, n))]
        elif preset in {"heavy-hex", "heavy_hex", "heavyhex"}:
            # Small heavy-hex-like template (toy fragment); will be truncated to q if needed
            # Node indices 0..6
            hh = [
                [0, 1], [1, 2],        # line
                [1, 3],                # branch to a hub
                [3, 4], [3, 5],        # hub connections
                [5, 6],                # tail
            ]
            n = int(args.q)
            coupling_edges = [e for e in hh if max(e) < n]
            basis = ["rz", "sx", "x", "cx"]
        else:
            print(f"Warning: unknown preset '{args.preset}', ignoring.", file=sys.stderr)

    # Explicit --coupling overrides preset coupling
    if args.coupling:
        try:
            coupling_edges = json.loads(args.coupling)
        except Exception as ex:
            raise SystemExit(f"Failed to parse --coupling JSON: {ex}")

    cm = CouplingMap(coupling_edges) if coupling_edges else None

    from qiskit import transpile as q_transpile
    tqc = q_transpile(qc, basis_gates=basis, coupling_map=cm, optimization_level=1)
    s0 = summarize_circuit(qc)
    s1 = summarize_circuit(tqc)
    report = {
        "pre": s0,
        "post": s1,
        "basis": basis,
        "coupling": coupling_edges,
        "preset": args.preset or None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

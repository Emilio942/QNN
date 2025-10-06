#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn.io import load_tensor_spec
from qnn.models import save_params


def bucket_means(w: List[float], q: int) -> List[float]:
    n = len(w)
    if q <= 0:
        return []
    means: List[float] = []
    for j in range(q):
        a = (j * n) // q
        b = ((j + 1) * n) // q
        if a >= b:
            means.append(0.0)
        else:
            seg = w[a:b]
            means.append(sum(seg) / len(seg))
    return means


def map_linear_to_qparams(w: List[float], b: float, q: int, L: int, scale: float = 1.0) -> List[float]:
    """Map a linear/logistic model (w,b) to QNN re-upload params.
    Strategy: compute q bucket means from w; scale to angles; replicate across L; add a small bias offset in first layer.
    """
    import math
    base = bucket_means(w, q)
    max_abs = max(1e-9, max(abs(x) for x in base))
    normed = [x / max_abs for x in base]
    # Map to [-pi/2, pi/2] by default, scaled by 'scale'
    mapped = [scale * (math.pi / 2.0) * x for x in normed]
    # Add bias as a uniform offset to first layer (small fraction)
    bias_off = scale * (math.pi / 8.0) * (b / max_abs)
    layers: List[float] = []
    for l in range(L):
        layer = [m + (bias_off if l == 0 else 0.0) for m in mapped]
        layers.extend(layer)
    # Clip to [-pi, pi]
    clipped = [max(-math.pi, min(math.pi, v)) for v in layers]
    return clipped


def _infer_d_from_spec(spec_path: str) -> int:
    spec = load_tensor_spec(spec_path)
    shape = spec.get("shape", {})
    layout = shape.get("layout", "d")
    if layout == "d":
        return int(shape.get("d"))
    # Fallback for HxWxC
    H = int(shape.get("H", 1))
    W = int(shape.get("W", 1))
    C = int(shape.get("C", 1))
    return H * W * C


def _load_weights_any(weights_path: str) -> Tuple[List[float], float, str]:
    """Load weights/bias from various formats.
    Supported:
      - JSON with keys 'w' (list) and optional 'b'
      - CSV (one row or one column of numbers); bias via a header 'b' or last value if --bias-last used (not implemented here)
      - NPY/NPZ (vector) — if NPZ, tries keys ['w', 'weights', 'arr_0']
      - scikit-learn pickle (LogisticRegression/SGDClassifier/LinearSVC) — when file endswith .pkl/.pickle/.joblib
    Returns (w, b, source_desc)
    """
    p = Path(weights_path)
    ext = p.suffix.lower()
    # scikit-learn pickle
    if ext in {".pkl", ".pickle", ".joblib"}:
        try:
            try:
                import joblib  # type: ignore
                model = joblib.load(str(p))
            except Exception:
                import pickle
                with open(p, "rb") as f:
                    model = pickle.load(f)
        except Exception as e:
            raise SystemExit(f"Failed to load sklearn pickle: {e}")
        # Try attributes in order
        w = None
        b = 0.0
        for attr in ["coef_", "weights_", "coefs_"]:
            if hasattr(model, attr):
                arr = getattr(model, attr)
                try:
                    import numpy as np  # noqa
                except Exception:
                    pass
                try:
                    import numpy as np
                    arr = np.asarray(arr).reshape(-1)
                    w = [float(x) for x in arr.tolist()]
                except Exception:
                    # Fallback: iterate
                    flat = []
                    try:
                        for row in arr:
                            try:
                                flat.extend(list(map(float, row)))
                            except TypeError:
                                flat.append(float(row))
                    except TypeError:
                        flat = [float(arr)]
                    w = flat
                break
        for battr in ["intercept_", "bias_", "intercepts_"]:
            if hasattr(model, battr):
                try:
                    import numpy as np
                    b_arr = np.asarray(getattr(model, battr)).reshape(-1)
                    b = float(b_arr[0])
                except Exception:
                    try:
                        b = float(getattr(model, battr)[0])
                    except Exception:
                        try:
                            b = float(getattr(model, battr))
                        except Exception:
                            pass
                break
        if not w:
            raise SystemExit("Loaded sklearn model but couldn't find weight vector (coef_/weights_)")
        return w, b, f"sklearn:{type(model).__name__}"

    if ext == ".json":
        data = json.loads(p.read_text())
        w = list(map(float, data.get("w", [])))
        b = float(data.get("b", 0.0))
        if not w:
            raise SystemExit("weights JSON must contain key 'w' with a non-empty list")
        return w, b, "json"

    if ext == ".csv":
        import csv
        vals: List[float] = []
        with open(p, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                raise SystemExit("CSV is empty")
            # If header contains 'w' or numeric-only
            try:
                # Flatten all numeric cells
                for row in rows:
                    for cell in row:
                        if cell is None or cell == "":
                            continue
                        vals.append(float(cell))
            except ValueError:
                # Skip header row if not numeric
                for row in rows[1:]:
                    for cell in row:
                        if cell is None or cell == "":
                            continue
                        vals.append(float(cell))
        if not vals:
            raise SystemExit("CSV contained no numeric values")
        return vals, 0.0, "csv"

    if ext in {".npy", ".npz"}:
        try:
            import numpy as np
        except Exception as e:
            raise SystemExit(f"NumPy required to load {ext} files: {e}")
        if ext == ".npy":
            arr = np.load(str(p))
            w = np.asarray(arr, dtype=float).reshape(-1).tolist()
            return w, 0.0, "npy"
        else:
            data = np.load(str(p))
            for key in ["w", "weights", "arr_0"]:
                if key in data:
                    arr = data[key]
                    w = np.asarray(arr, dtype=float).reshape(-1).tolist()
                    return w, 0.0, f"npz:{key}"
            # Fallback: first array
            if len(list(data.keys())):
                key0 = list(data.keys())[0]
                arr = data[key0]
                w = np.asarray(arr, dtype=float).reshape(-1).tolist()
                return w, 0.0, f"npz:{key0}"
            raise SystemExit("NPZ file contained no arrays")

    raise SystemExit(f"Unsupported weights format for: {weights_path}")


def main():
    ap = argparse.ArgumentParser(description="Initialize QNN parameters from a small linear/logistic model (approximation). Not a 1:1 converter.")
    ap.add_argument("--weights", required=True, help="Path to weights (JSON/CSV/NPY/NPZ or sklearn pickle: .pkl/.pickle/.joblib)")
    ap.add_argument("--spec", default=str(ROOT / "specs" / "tensor_spec.yaml"), help="Tensor spec path (to get d)")
    ap.add_argument("--q", type=int, required=True, help="Qubits for QNN")
    ap.add_argument("--L", type=int, required=True, help="Re-upload layers for QNN")
    ap.add_argument("--scale", type=float, default=1.0, help="Scaling factor for angle mapping")
    ap.add_argument("--out", required=True, help="Output params JSON path")
    args = ap.parse_args()

    d = _infer_d_from_spec(args.spec)
    w, b, source = _load_weights_any(args.weights)
    if len(w) != d:
        print(f"Warning: weight length {len(w)} != spec d {d}; will still bucket-average.", file=sys.stderr)

    theta = map_linear_to_qparams(w, b, int(args.q), int(args.L), float(args.scale))
    save_params(args.out, theta, int(args.q), int(args.L))
    print(json.dumps({
        "saved_params": args.out,
        "q": int(args.q),
        "L": int(args.L),
        "theta_len": len(theta),
        "source": source,
        "note": "Initialization only; not a direct NN→QNN conversion"
    }, indent=2))


if __name__ == "__main__":
    main()

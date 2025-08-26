from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import json
import csv


def load_vectors_labels(path: str | Path) -> Tuple[List[List[float]], List[int]]:
    """Load dataset with features and labels from JSON or CSV.

    JSON format: {"X": [[...], ...], "y": [1,-1,...]}
    CSV format: each row is features...,label (label in the last column)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        X = data.get("X")
        y = data.get("y")
        if X is None or y is None:
            raise ValueError("JSON must contain keys 'X' and 'y'")
        return X, y
    # CSV
    X: List[List[float]] = []
    y: List[int] = []
    with p.open() as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            *feat, lab = row
            X.append([float(v) for v in feat])
            y.append(int(lab))
    return X, y

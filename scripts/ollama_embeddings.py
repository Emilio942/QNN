#!/usr/bin/env python3
from __future__ import annotations

"""
Fetch text embeddings from a local Ollama server and export a dataset for qnn-train/qnn-predict.

Inputs supported:
- .txt: one text per line; optionally provide --labels .txt with one label per line (1/-1)
- .json: either {"texts": [...]}, {"records": [{"text": "...", "label": 1}, ...]}, or {"X": [...], "y": [...]}
- .csv: expects columns 'text' and optional 'label' unless overridden by --text-col/--label-col

Outputs:
- .json: {"X": [[...], ...], "y": [1,-1,...]} (y omitted if no labels)
- .csv: rows of features with last column as label if available

Note: Ensure specs/tensor_spec.yaml 'd' matches the embedding dimension produced by your model.
"""

from pathlib import Path
import sys
import json
import argparse
import csv
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_texts_labels(path: Path, labels_path: Optional[Path], text_col: str, label_col: Optional[str]) -> Tuple[List[str], Optional[List[int]]]:
    texts: List[str] = []
    labels: Optional[List[int]] = None
    if path.suffix.lower() == ".txt":
        texts = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if labels_path:
            lab = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            labels = [int(x) for x in lab]
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if "texts" in data:
                texts = list(map(str, data["texts"]))
            elif "records" in data:
                recs = list(data["records"])
                texts = [str(r.get("text", "")) for r in recs]
                if any("label" in r for r in recs):
                    labels = [int(r.get("label", 0)) for r in recs]
            elif "X" in data and isinstance(data["X"], list) and all(isinstance(t, str) for t in data["X"]):
                texts = list(map(str, data["X"]))
                if "y" in data:
                    labels = [int(v) for v in data["y"]]
            else:
                raise SystemExit("Unsupported JSON structure. Use {'texts': [...] } or {'records': [{'text','label'}]} or {'X': [...], 'y': [...]}.")
        else:
            raise SystemExit("JSON root must be an object.")
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if text_col not in r.fieldnames:
                raise SystemExit(f"CSV missing text column '{text_col}'. Fields: {r.fieldnames}")
            texts = [str(row[text_col]) for row in r]
        if label_col:
            with path.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                if label_col not in r.fieldnames:
                    raise SystemExit(f"CSV missing label column '{label_col}'. Fields: {r.fieldnames}")
                labels = [int(row[label_col]) for row in r]
    else:
        raise SystemExit("Unsupported input extension. Use .txt, .json, or .csv")
    return texts, labels


def _fetch_embedding_ollama(model: str, text: str, url: str) -> List[float]:
    import requests
    resp = requests.post(url, json={"model": model, "prompt": text})
    resp.raise_for_status()
    data = resp.json()
    if "embedding" not in data:
        raise RuntimeError(f"Ollama response missing 'embedding': keys={list(data.keys())}")
    emb = data["embedding"]
    return list(map(float, emb))


def _save_json(out: Path, X: List[List[float]], y: Optional[List[int]]):
    payload = {"X": X}
    if y is not None:
        payload["y"] = y
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))


def _save_csv(out: Path, X: List[List[float]], y: Optional[List[int]]):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        for i, vec in enumerate(X):
            row = list(map(float, vec))
            if y is not None:
                row.append(int(y[i]))
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Fetch embeddings from a local Ollama server and export dataset")
    p.add_argument("--in", dest="inp", required=True, help="Input file (.txt/.json/.csv)")
    p.add_argument("--labels", help="Optional labels file (.txt), one label per line")
    p.add_argument("--text-col", default="text", help="CSV text column name (default: text)")
    p.add_argument("--label-col", default=None, help="CSV label column name (default: None)")
    p.add_argument("--model", default="nomic-embed-text", help="Ollama embedding model name")
    p.add_argument("--url", default="http://localhost:11434/api/embeddings", help="Ollama embeddings endpoint")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of texts")
    p.add_argument("--out", required=True, help="Output path (.json or .csv)")
    args = p.parse_args()

    inp = Path(args.inp)
    labels_path = Path(args.labels) if args.labels else None
    texts, y = _load_texts_labels(inp, labels_path, args.text_col, args.label_col)
    if args.limit is not None:
        texts = texts[: int(args.limit)]
        if y is not None:
            y = y[: int(args.limit)]

    X: List[List[float]] = []
    dim: Optional[int] = None
    for i, t in enumerate(texts):
        emb = _fetch_embedding_ollama(args.model, t, args.url)
        if dim is None:
            dim = len(emb)
        X.append(list(map(float, emb)))
        if (i + 1) % 10 == 0:
            print(f"Fetched {i+1}/{len(texts)} embeddings...", file=sys.stderr)

    print(json.dumps({"n": len(X), "d": dim, "labels": (y is not None)}, indent=2))

    out = Path(args.out)
    if out.suffix.lower() == ".json":
        _save_json(out, X, y)
    elif out.suffix.lower() == ".csv":
        _save_csv(out, X, y)
    else:
        raise SystemExit("Output must be .json or .csv")


if __name__ == "__main__":
    main()

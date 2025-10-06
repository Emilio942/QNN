# QNN Tensor → Quantum State (End-to-End)

Minimal scaffolding to specify a tensor, preprocess it, pick an encoding, build circuits, and simulate measurements with/without noise.

## Setup

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demos

- Spec check:
  ```zsh
  python scripts/smoke_validate_spec.py
  ```
- Preprocess demo:
  ```zsh
  python scripts/demo_preprocess.py
  ```
- Angle sim (ideal):
  ```zsh
  python scripts/demo_angle_sim.py
  ```
- Phase sim (ideal):
  ```zsh
  python scripts/demo_phase_sim.py
  ```
- Image (4x4x1) angle demo:
  ```zsh
  python scripts/demo_image_angle.py
  ```
- Amplitude sim (StatePreparation):
  ```zsh
  python scripts/demo_amplitude_sim.py
  ```
- Noisy angle sim (Aer):
  ```zsh
  python scripts/demo_noise_angle.py
  ```
- VQA (SPSA) demo:
  ```zsh
  python scripts/demo_vqa_spsa.py
  ```
- Data re-upload classifier (toy):
  ```zsh
  python scripts/demo_reupload_classifier.py
  ```
- Train re-upload classifier (toy):
  ```zsh
  python scripts/train_reupload_classifier.py --steps 30 --batch 32 --q 4 --L 2 --out reports/reupload_params.json --seed 0
  ```
  As console command:
  ```zsh
  qnn-train --steps 30 --batch 32 --q 4 --L 2 --out reports/reupload_params.json --seed 0
  ```
  Train with your dataset (CSV with last column label, or JSON {"X": [...], "y": [...]})
  ```zsh
  qnn-train --data data/train.csv --val-data data/val.csv --out reports/reupload_params.json --seed 0
  ```
  Predict with saved params:
  ```zsh
  python scripts/predict_reupload_classifier.py --params reports/reupload_params.json --n 5 --seed 0 --export reports/predict.json
  # or with your own JSON vectors file
  python scripts/predict_reupload_classifier.py --params reports/reupload_params.json --input path/to/vectors.json --export reports/predict.csv
  ```
  As console command:
  ```zsh
  qnn-predict --params reports/reupload_params.json --n 5 --seed 0 --export reports/predict.json
  ```

## Evaluation

Evaluate saved params on a dataset (CSV or JSON). CSV expects rows of features with the last column as label; JSON expects {"X": [[...], ...], "y": [1,-1,...]}.

```zsh
# Using toy fallback (no data file), with export
python scripts/eval_classifier.py --params reports/reupload_params.json --seed 0 --export reports/eval.json

# With a dataset file
python scripts/eval_classifier.py --params reports/reupload_params.json --data path/to/data.csv --export reports/eval.csv

# Via config
python scripts/eval_classifier.py --config configs/example.yaml
```

Optional plots:

```zsh
# Precision-Recall curve
python scripts/eval_classifier.py --params reports/reupload_params.json --seed 0 --plot reports/pr.png

# ROC curve
python scripts/eval_classifier.py --params reports/reupload_params.json --seed 0 --roc-plot reports/roc.png
```

Both plots can be produced in one run by supplying both flags.

## Configs

You can keep common settings in a YAML/JSON file, e.g. `configs/example.yaml`:

```yaml
spec: specs/tensor_spec.yaml
train:
  steps: 50
  batch: 32
  q: 4
  L: 2
  noise: false
  out: reports/reupload_params.json
predict:
  params: reports/reupload_params.json
  n: 5
```
Use the values by passing them as CLI args (config loader utility is available at `qnn/config.py`; integrate as needed).
Or run with a config file directly:

```zsh
qnn-train --config configs/example.yaml
qnn-predict --config configs/example.yaml
```

## Hardware transpile summary

Summarize pre/post transpilation metrics under basis/coupling constraints.

```zsh
# Linear chain of 3 qubits (preset)
qnn-hw --q 3 --L 1 --preset linear3 --out reports/hw_linear3.json

# Ring of 4 qubits (preset)
qnn-hw --q 4 --L 1 --preset ring4 --out reports/hw_ring4.json

# Heavy-hex toy template (truncated to q)
qnn-hw --q 5 --L 1 --preset heavy-hex --out reports/hw_hh.json

# Custom basis/coupling (explicit overrides preset)
qnn-hw --q 3 --L 1 --basis rz sx x cx --coupling "[[0,1],[1,2]]" --out reports/hw_custom.json
```

Output JSON contains: pre and post summaries, basis, coupling, and the used preset.

## Embeddings via Ollama → Dataset

Hole dir Embeddings von einer lokalen Ollama‑Instanz und exportiere ein Dataset für Training/Eval:

```zsh
# Textdatei: eine Zeile = ein Text; optional Labels als separate .txt (1/-1)
qnn-embed --in texts.txt --labels labels.txt --model nomic-embed-text --out reports/embeds.json

# CSV mit Spalten 'text' und 'label'
qnn-embed --in data.csv --text-col text --label-col label --out reports/embeds.csv

# Danach Spec-Dimension (d) auf Embedding-Länge setzen und trainieren
qnn-train --data reports/embeds.csv --out reports/params_from_embeds.json --q 4 --L 2 --seed 0
```

Voraussetzung: Ollama läuft lokal und stellt /api/embeddings bereit (Standardport 11434).

## CI artifacts

The CI builds a small model, runs hardware transpile summary, and produces plots.
Artifacts include:
- reports/hw_ci.json — transpile metrics
- reports/pr_ci.png — PR curve plot
- (when used) reports/roc_ci.png — ROC curve plot

## Intent Router (Hybrid)

Klassifiziere den Intent mit QNN und antworte direkt (positiv) oder delegiere an ein LLM (negativ):

```zsh
# Offline-Embedding (d=16, Standard-Spec)
qnn-intent --text "hi" --params reports/greet_params.json --offline

# Mit Ollama-Embeddings (nutze passende Spec mit d=768)
qnn-intent --text "hi" --params reports/ollama_greet_params.json --spec specs/tensor_spec_ollama.yaml \
  --delegate-negative --model-generate llama3
```

## Auto-Spec nach Datensatz

Erzeuge eine Vektor-Spec mit d aus einem bestehenden Dataset:

```zsh
qnn-make-spec --data reports/embeds.json --out specs/tensor_spec_auto.yaml
```

## NN → QNN (Initialisierung)

Hinweis: Das ist eine Initialisierung, kein vollwertiger NN→QNN‑Konverter. Ziel: sinnvolle Start‑Parameter aus einem kleinen linearen/logistischen Modell ableiten.

Unterstützte Eingaben: JSON, CSV, NPY/NPZ, sowie scikit‑learn Pickle (.pkl/.pickle/.joblib). JSON erwartet `{ "w": [...], "b": <optional> }`.

```zsh
# Schnellstart mit Beispielgewichten (d=16)
qnn-nn2qnn --weights configs/weights.example.json --spec specs/tensor_spec.yaml --q 4 --L 2 --out reports/params_init.json

# Eigene Gewichte
qnn-nn2qnn --weights weights.json --spec specs/tensor_spec.yaml --q 4 --L 2 --out reports/params_init.json
```

## Tests

```zsh
python scripts/test_preprocess.py
```

```

## Spec
Fill `specs/tensor_spec.yaml` (dtype, shape, indexing, encoding target, scaling). See `aufgabeliste.md` item 1 for the detailed checklist.

## License

MIT License — see `LICENSE`.

## Documentation

- **[References](docs/REFERENCES.md)** — wissenschaftliche Quellen und verwandte Arbeiten (QML/VQAs, Data Re-uploading, SPSA, Qiskit, PR/ROC, Embeddings)
- **[Problem Formulation](docs/PROBLEM_FORMULATION.md)** — mathematische Problemstellungen, Lösungsframework und Implementierungsstatus


## License

MIT License — see `LICENSE`.

## Documentation

- **[References](docs/REFERENCES.md)** — wissenschaftliche Quellen und verwandte Arbeiten (QML/VQAs, Data Re-uploading, SPSA, Qiskit, PR/ROC, Embeddings)
- **[Problem Formulation](docs/PROBLEM_FORMULATION.md)** — mathematische Problemstellungen, Lösungsframework und Implementierungsstatus

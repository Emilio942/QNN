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

## Tests

```zsh
python scripts/test_preprocess.py
```

## Spec
Fill `specs/tensor_spec.yaml` (dtype, shape, indexing, encoding target, scaling). See `aufgabeliste.md` item 1 for the detailed checklist.

## License

MIT License — see `LICENSE`.

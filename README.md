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

## Tests

```zsh
python scripts/test_preprocess.py
```

## Spec
Fill `specs/tensor_spec.yaml` (dtype, shape, indexing, encoding target, scaling). See `aufgabeliste.md` item 1 for the detailed checklist.

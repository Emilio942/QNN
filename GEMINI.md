# Gemini Code Understanding Report: QNN Pipeline

## Project Overview

This project, `qnn-pipeline`, is a Python-based framework for exploring and simulating Quantum Neural Networks (QNNs). It provides a set of tools and demos for encoding classical tensor data into quantum states, building and simulating quantum circuits, and training and evaluating QNN models. The project uses the Qiskit library for quantum circuit simulation and offers various command-line utilities for tasks like training, prediction, hardware transpilation analysis, and more. It also integrates with Ollama for generating embeddings from text.

## Building and Running

### Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

### Running Demos and Tests

The project includes a script to run all demos and basic tests:

```bash
python scripts/run_all.py
```

or via the command-line tool:

```bash
qnn-run-all
```

The CI workflow in `.github/workflows/ci.yml` provides a comprehensive set of commands for testing and validation. The main testing command is:

```bash
python -m pytest -q
```

### Command-Line Tools

The project provides several command-line tools for interacting with the QNN pipeline:

*   `qnn-train`: Train a data re-uploading classifier.
*   `qnn-predict`: Make predictions with a trained classifier.
*   `qnn-hw`: Summarize pre/post transpilation metrics for hardware.
*   `qnn-embed`: Get embeddings from an Ollama instance.
*   `qnn-intent`: Classify intent with QNN and delegate to an LLM.
*   `qnn-make-spec`: Generate a vector spec from a dataset.
*   `qnn-nn2qnn`: Initialize QNN parameters from a classical neural network.

For detailed usage of each command, refer to the `README.md` file.

## Development Conventions

*   **Configuration:** Project configuration is managed through YAML files (e.g., `configs/example.yaml`). The `qnn.config` module provides utilities for loading configurations.
*   **Dependencies:** Project dependencies are listed in `requirements.txt`.
*   **Command-Line Interfaces:** CLIs are defined in `pyproject.toml` and implemented in the `scripts` directory.
*   **Testing:** The project uses `pytest` for unit testing (see the `tests` directory). The CI configuration in `.github/workflows/ci.yml` is a good reference for testing procedures.
*   **Code Style:** The code follows standard Python conventions.

This report provides a high-level overview of the `qnn-pipeline` project. For more detailed information, please refer to the `README.md` file and the source code.

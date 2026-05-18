# From Syntax to Circuits: QNLP Tutorial

Physics 14N project exploring a toy Quantum Natural Language Processing
(QNLP) sentiment classifier.

The project turns small English sentences into grammar diagrams, maps those
diagrams to toy parameterized quantum circuits, trains a binary classifier, and
exports visual assets for a poster presentation.

## Deliverables

- `qnlp_tutorial.ipynb`: interactive tutorial and end-to-end simulation.
- `data/toy_sentiment.csv`: constrained positive/negative sentence dataset.
- `src/qnlp_tutorial/`: reusable helpers for data loading, parsing, circuit
  construction, training, evaluation, and visualization.
- `poster_assets/`: generated diagrams, plots, and poster outline.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,quantum]"
```

The core tutorial code runs with NumPy, pandas, and matplotlib. Qiskit is used
when available to render circuit diagrams. `lambeq` is listed as an optional
dependency because its parser stack can be more sensitive to platform and model
setup; this tutorial includes a deterministic constrained-grammar path for the
small dataset so the project remains reproducible on Apple Silicon.

## Run

Open the notebook:

```bash
jupyter notebook qnlp_tutorial.ipynb
```

Or run a smoke test from the command line:

```bash
python -m qnlp_tutorial.demo
```


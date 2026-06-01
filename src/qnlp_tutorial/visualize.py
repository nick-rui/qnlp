"""Visualization helpers for the notebook and poster assets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qnlp_tutorial.grammar import GrammarDiagram
from qnlp_tutorial.quantum import ToyCircuit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSTER_ASSETS = PROJECT_ROOT / "poster_assets"


def ensure_assets_dir(path: str | Path = POSTER_ASSETS) -> Path:
    assets_dir = Path(path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


def save_text_asset(text: str, filename: str, assets_dir: str | Path = POSTER_ASSETS) -> Path:
    path = ensure_assets_dir(assets_dir) / filename
    path.write_text(text, encoding="utf-8")
    return path


def save_grammar_diagram(
    diagram: GrammarDiagram,
    filename: str = "grammar_diagram.png",
    assets_dir: str | Path = POSTER_ASSETS,
) -> Path:
    path = ensure_assets_dir(assets_dir) / filename
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(
        0.02,
        0.92,
        "Grammar to String Diagram",
        fontsize=18,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(0.02, 0.72, diagram.ascii_tree(), fontsize=13, family="monospace", transform=ax.transAxes)
    ax.text(
        0.56,
        0.72,
        diagram.string_diagram_text(),
        fontsize=10.5,
        family="monospace",
        va="top",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_circuit_diagram(
    circuit: ToyCircuit,
    filename: str = "toy_circuit.png",
    assets_dir: str | Path = POSTER_ASSETS,
) -> Path:
    path = ensure_assets_dir(assets_dir) / filename
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    ax.text(0.02, 0.86, "DisCoCat-Inspired Two-Qubit Circuit", fontsize=17, fontweight="bold")
    ax.text(0.02, 0.57, circuit.ascii_diagram(), fontsize=13, family="monospace")
    ax.text(0.02, 0.20, circuit.describe(), fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_loss_curve(
    history: list[dict[str, float]],
    filename: str = "loss_curve.png",
    assets_dir: str | Path = POSTER_ASSETS,
) -> Path:
    path = ensure_assets_dir(assets_dir) / filename
    history_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(history_df["epoch"], history_df["loss"], color="#2c5f8a", linewidth=1.6)
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Validation loss", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_poster_outline(
    train_accuracy: float,
    test_accuracy: float,
    assets_dir: str | Path = POSTER_ASSETS,
) -> Path:
    text = f"""# Poster Outline

## Section 1: The Concept
Classical NLP usually learns vector embeddings from text alone. QNLP starts
from grammar: nouns become states, grammatical reductions wire meanings
together, and parameterized circuit operations compose sentence meaning.

## Section 2: Visualizing Grammar
Use `grammar_diagram.png` beside `toy_circuit.png` to show one sentence moving
from syntax into a two-qubit circuit.

## Section 3: Results
Use `loss_curve.png` and report:

- Training accuracy: {train_accuracy:.1%}
- Test accuracy: {test_accuracy:.1%}

Caption: even a tiny circuit-inspired model can learn the controlled semantic
distinction when grammar determines how word parameters compose.
"""
    return save_text_asset(text, "poster_outline.md", assets_dir=assets_dir)


from __future__ import annotations

import json
from pathlib import Path

from qnlp_tutorial.data import load_dataset, train_test_split
from qnlp_tutorial.grammar import parse_sentence
from qnlp_tutorial.quantum import ToyQNLPClassifier
from qnlp_tutorial.visualize import (
    save_circuit_diagram,
    save_grammar_diagram,
    save_loss_curve,
    write_poster_outline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_is_balanced_and_split() -> None:
    dataset = load_dataset()
    train, test = train_test_split(dataset)

    assert len(dataset) == 72
    assert len(train) == 60
    assert len(test) == 12
    assert dataset["label"].value_counts().to_dict() == {1: 36, 0: 36}


def test_controlled_grammar_parses_transitive_sentence() -> None:
    diagram = parse_sentence("the helpful agent fixes the bug")

    assert diagram.subject == "agent"
    assert diagram.verb == "fixes"
    assert diagram.object_ == "bug"
    assert diagram.roles["subject_modifier"] == "helpful"
    assert "cups contract" in diagram.string_diagram_text()


def test_model_trains_and_predicts_holdout_split() -> None:
    dataset = load_dataset()
    train, test = train_test_split(dataset)
    model = ToyQNLPClassifier.from_dataset(dataset)

    initial_loss = model.loss(train)
    history = model.fit(train, epochs=50, learning_rate=0.35)

    assert history[-1]["loss"] < initial_loss
    assert model.accuracy(train) >= 0.95
    assert model.accuracy(test) >= 0.90


def test_visual_asset_generation(tmp_path: Path) -> None:
    dataset = load_dataset()
    train, test = train_test_split(dataset)
    model = ToyQNLPClassifier.from_dataset(dataset)
    history = model.fit(train, epochs=5)
    diagram = parse_sentence("the helpful agent fixes the bug")

    paths = [
        save_grammar_diagram(diagram, assets_dir=tmp_path),
        save_circuit_diagram(model.circuit_for(diagram), assets_dir=tmp_path),
        save_loss_curve(history, assets_dir=tmp_path),
        write_poster_outline(model.accuracy(train), model.accuracy(test), assets_dir=tmp_path),
    ]

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_notebook_is_valid_json() -> None:
    notebook = json.loads((PROJECT_ROOT / "qnlp_tutorial.ipynb").read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "markdown" for cell in notebook["cells"])
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])


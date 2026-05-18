"""Command-line smoke test for the QNLP tutorial pipeline."""

from __future__ import annotations

from qnlp_tutorial.data import load_dataset, train_test_split
from qnlp_tutorial.grammar import parse_sentence
from qnlp_tutorial.quantum import ToyQNLPClassifier
from qnlp_tutorial.visualize import (
    save_circuit_diagram,
    save_grammar_diagram,
    save_loss_curve,
    write_poster_outline,
)


def run_demo(epochs: int = 50) -> dict[str, float]:
    dataset = load_dataset()
    train, test = train_test_split(dataset)

    model = ToyQNLPClassifier.from_dataset(dataset)
    example = parse_sentence(train.iloc[0]["sentence"])
    history = model.fit(train, epochs=epochs)

    train_accuracy = model.accuracy(train)
    test_accuracy = model.accuracy(test)

    save_grammar_diagram(example)
    save_circuit_diagram(model.circuit_for(example))
    save_loss_curve(history)
    write_poster_outline(train_accuracy, test_accuracy)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "final_loss": history[-1]["loss"],
    }


def main() -> None:
    metrics = run_demo()
    print("QNLP tutorial smoke test complete")
    print(f"train accuracy: {metrics['train_accuracy']:.3f}")
    print(f"test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"final loss: {metrics['final_loss']:.3f}")


if __name__ == "__main__":
    main()


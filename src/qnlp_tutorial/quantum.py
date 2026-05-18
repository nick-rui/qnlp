"""Toy DisCoCat-inspired circuit and classifier utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np
import pandas as pd

from qnlp_tutorial.grammar import GrammarDiagram, parse_sentence


ROLE_WEIGHTS = {
    "subject_modifier": 1.25,
    "subject": 0.45,
    "verb": 1.4,
    "object_modifier": 0.9,
    "object": 0.35,
}


@dataclass(frozen=True)
class ToyCircuit:
    """A small two-qubit circuit template induced by a grammar diagram."""

    diagram: GrammarDiagram
    subject_angle: float
    predicate_angle: float
    decision_angle: float

    def ascii_diagram(self) -> str:
        """Return a text circuit diagram that works without Qiskit."""
        return (
            "q_subject: |0> --Ry(subject phrase)--*----------------\n"
            "                                      |\n"
            "q_sentence:|0> --Ry(predicate)------X--Ry(decision)--measure"
        )

    def qiskit_circuit(self):
        """Build a Qiskit circuit when qiskit is installed."""
        try:
            from qiskit import QuantumCircuit
        except ImportError as exc:
            raise ImportError("Install qiskit to render this circuit with Qiskit.") from exc

        circuit = QuantumCircuit(2, 1)
        circuit.ry(self.subject_angle, 0)
        circuit.ry(self.predicate_angle, 1)
        circuit.cx(0, 1)
        circuit.ry(self.decision_angle, 1)
        circuit.measure(1, 0)
        return circuit

    def describe(self) -> str:
        """Human-readable circuit parameters."""
        return (
            f"subject Ry angle: {self.subject_angle:.3f}\n"
            f"predicate Ry angle: {self.predicate_angle:.3f}\n"
            f"decision Ry angle: {self.decision_angle:.3f}"
        )


class ToyQNLPClassifier:
    """Train word angles for a controlled QNLP-style sentiment task.

    Each content word owns one trainable scalar. The grammar chooses role
    weights for those scalars, producing a sentence-level angle. The model maps
    that angle to the probability of measuring `1` on the sentence qubit.
    """

    def __init__(self, vocabulary: list[str] | tuple[str, ...], seed: int = 7) -> None:
        self.vocabulary = tuple(sorted(set(vocabulary)))
        rng = np.random.default_rng(seed)
        self.parameters = {
            word: float(rng.normal(loc=0.0, scale=0.05)) for word in self.vocabulary
        }
        self.bias = 0.0

    @classmethod
    def from_dataset(cls, dataset: pd.DataFrame, seed: int = 7) -> "ToyQNLPClassifier":
        words: set[str] = set()
        for sentence in dataset["sentence"]:
            words.update(parse_sentence(sentence).content_words)
        return cls(sorted(words), seed=seed)

    def features(self, diagram: GrammarDiagram) -> dict[str, float]:
        """Return role-weighted content-word features for a parsed sentence."""
        features: dict[str, float] = {}
        roles = diagram.roles
        for role, word in roles.items():
            features[word] = features.get(word, 0.0) + ROLE_WEIGHTS[role]
        return features

    def score(self, diagram: GrammarDiagram) -> float:
        """Compute the sentence angle/logit from trainable word parameters."""
        score = self.bias
        for word, weight in self.features(diagram).items():
            score += weight * self.parameters.get(word, 0.0)
        return float(score)

    def predict_proba(self, diagram_or_sentence: GrammarDiagram | str) -> float:
        """Return P(label=1), interpreted as the measured sentence qubit probability."""
        diagram = (
            parse_sentence(diagram_or_sentence)
            if isinstance(diagram_or_sentence, str)
            else diagram_or_sentence
        )
        logit = self.score(diagram)
        if logit >= 0:
            return 1.0 / (1.0 + exp(-logit))
        exp_logit = exp(logit)
        return exp_logit / (1.0 + exp_logit)

    def predict(self, diagram_or_sentence: GrammarDiagram | str) -> int:
        return int(self.predict_proba(diagram_or_sentence) >= 0.5)

    def circuit_for(self, diagram_or_sentence: GrammarDiagram | str) -> ToyCircuit:
        """Construct the two-qubit circuit template for a sentence."""
        diagram = (
            parse_sentence(diagram_or_sentence)
            if isinstance(diagram_or_sentence, str)
            else diagram_or_sentence
        )
        subject_angle = 0.0
        predicate_angle = 0.0
        if diagram.adjective:
            subject_angle += self.parameters.get(diagram.adjective, 0.0)
        subject_angle += self.parameters.get(diagram.subject, 0.0)
        predicate_angle += self.parameters.get(diagram.verb, 0.0)
        if diagram.object_adjective:
            predicate_angle += self.parameters.get(diagram.object_adjective, 0.0)
        if diagram.object_:
            predicate_angle += self.parameters.get(diagram.object_, 0.0)
        return ToyCircuit(
            diagram=diagram,
            subject_angle=subject_angle,
            predicate_angle=predicate_angle,
            decision_angle=self.score(diagram),
        )

    def loss(self, rows: pd.DataFrame) -> float:
        """Binary cross entropy over a DataFrame of rows."""
        losses = []
        eps = 1e-8
        for row in rows.itertuples(index=False):
            probability = self.predict_proba(row.sentence)
            label = float(row.label)
            losses.append(-(label * np.log(probability + eps) + (1 - label) * np.log(1 - probability + eps)))
        return float(np.mean(losses))

    def accuracy(self, rows: pd.DataFrame) -> float:
        correct = 0
        for row in rows.itertuples(index=False):
            correct += int(self.predict(row.sentence) == int(row.label))
        return correct / len(rows)

    def fit(
        self,
        rows: pd.DataFrame,
        epochs: int = 50,
        learning_rate: float = 0.35,
    ) -> list[dict[str, float]]:
        """Train parameters with exact BCE gradients for the sentence-angle model."""
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            gradient = {word: 0.0 for word in self.vocabulary}
            bias_gradient = 0.0

            for row in rows.itertuples(index=False):
                diagram = parse_sentence(row.sentence)
                probability = self.predict_proba(diagram)
                error = probability - float(row.label)
                bias_gradient += error
                for word, weight in self.features(diagram).items():
                    gradient[word] += error * weight

            scale = 1.0 / len(rows)
            self.bias -= learning_rate * bias_gradient * scale
            for word in self.vocabulary:
                self.parameters[word] -= learning_rate * gradient[word] * scale

            history.append(
                {
                    "epoch": float(epoch),
                    "loss": self.loss(rows),
                    "accuracy": self.accuracy(rows),
                }
            )
        return history

    def prediction_table(self, rows: pd.DataFrame) -> pd.DataFrame:
        records = []
        for row in rows.itertuples(index=False):
            probability = self.predict_proba(row.sentence)
            records.append(
                {
                    "sentence": row.sentence,
                    "label": int(row.label),
                    "prediction": int(probability >= 0.5),
                    "p_positive": round(probability, 3),
                }
            )
        return pd.DataFrame.from_records(records)


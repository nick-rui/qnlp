"""Toy DisCoCat-inspired circuit and classifier utilities.

The classifier no longer reads sentiment off a classical sigmoid. Every
prediction is produced by *simulating* the two-qubit grammar circuit on a
statevector and measuring the sentence qubit, and parameters are trained with
the parameter-shift rule -- the same gradient technique real QNLP circuits use.
"""

from __future__ import annotations

from dataclasses import dataclass

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

# Operating point for the final decision rotation. Starting near pi/2 places the
# sentence qubit at the steepest part of the measurement curve, so word
# sentiments can push the measured probability up (positive) or down (negative).
READOUT_OFFSET = float(np.pi / 2)

_I2 = np.eye(2)
# Basis states are ordered |subject sentence> with index = 2*subject + sentence.
# CX with the subject qubit as control flips the sentence qubit only when the
# subject reads 1, i.e. it swaps basis states 10 (index 2) and 11 (index 3).
_CX_SUBJECT_SENTENCE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)


def _ry_matrix(theta: float) -> np.ndarray:
    """Single-qubit Ry(theta) rotation matrix."""
    half = theta / 2.0
    cos, sin = np.cos(half), np.sin(half)
    return np.array([[cos, -sin], [sin, cos]])


def simulate_sentence_probability(
    subject_angle: float, predicate_angle: float, decision_angle: float
) -> float:
    """Statevector-simulate the grammar circuit and return P(sentence qubit = 1).

    Gate order: Ry on the subject qubit, Ry on the sentence qubit, a CX that
    entangles subject -> sentence (the grammatical link), then the decision Ry
    on the sentence qubit. The returned value is the measured probability that
    the sentence qubit reads 1, interpreted as P(positive sentiment).
    """
    state = np.zeros(4)
    state[0] = 1.0  # both qubits start in |0>, i.e. basis state |00>
    state = np.kron(_ry_matrix(subject_angle), _I2) @ state
    state = np.kron(_I2, _ry_matrix(predicate_angle)) @ state
    state = _CX_SUBJECT_SENTENCE @ state
    state = np.kron(_I2, _ry_matrix(decision_angle)) @ state
    # The sentence qubit is the low bit; it reads 1 in basis states 01 and 11.
    return float(state[1] ** 2 + state[3] ** 2)


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
        """Return P(label=1) by simulating and measuring the sentence circuit.

        This is the genuine quantum readout: the angles built in `circuit_for`
        are fed through the statevector simulation and the sentence qubit is
        measured. The drawn circuit and the prediction now share one path.
        """
        circuit = self.circuit_for(diagram_or_sentence)
        return simulate_sentence_probability(
            circuit.subject_angle, circuit.predicate_angle, circuit.decision_angle
        )

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
            decision_angle=READOUT_OFFSET + self.score(diagram),
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
        """Train word angles by differentiating the simulated circuit.

        Gradients flow through the parameter-shift rule: each gate angle's
        derivative is (P(angle + pi/2) - P(angle - pi/2)) / 2 -- exact for an
        Ry rotation, and the standard way real QNLP circuits are trained. The
        chain rule then maps those per-angle gradients onto the word
        parameters, since each angle is a (role-weighted) sum of word dials.
        """
        shift = float(np.pi / 2)
        eps = 1e-7
        diagrams = [parse_sentence(row.sentence) for row in rows.itertuples(index=False)]
        labels = [float(row.label) for row in rows.itertuples(index=False)]

        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            gradient = {word: 0.0 for word in self.vocabulary}
            bias_gradient = 0.0

            for diagram, label in zip(diagrams, labels):
                circuit = self.circuit_for(diagram)
                sa, pa, da = circuit.subject_angle, circuit.predicate_angle, circuit.decision_angle

                probability = min(max(simulate_sentence_probability(sa, pa, da), eps), 1 - eps)

                # Parameter-shift derivatives of P w.r.t. each gate angle.
                d_subject = 0.5 * (
                    simulate_sentence_probability(sa + shift, pa, da)
                    - simulate_sentence_probability(sa - shift, pa, da)
                )
                d_predicate = 0.5 * (
                    simulate_sentence_probability(sa, pa + shift, da)
                    - simulate_sentence_probability(sa, pa - shift, da)
                )
                d_decision = 0.5 * (
                    simulate_sentence_probability(sa, pa, da + shift)
                    - simulate_sentence_probability(sa, pa, da - shift)
                )

                # d(BCE)/dP for this example.
                d_loss_d_prob = (probability - label) / (probability * (1 - probability))

                # Chain rule: which words feed which gate angle.
                subject_words = [w for w in (diagram.adjective, diagram.subject) if w]
                predicate_words = [
                    w for w in (diagram.verb, diagram.object_adjective, diagram.object_) if w
                ]
                decision_weights = self.features(diagram)  # word -> role weight

                touched = set(subject_words) | set(predicate_words) | set(decision_weights)
                for word in touched:
                    d_angle = (
                        d_subject * subject_words.count(word)
                        + d_predicate * predicate_words.count(word)
                        + d_decision * decision_weights.get(word, 0.0)
                    )
                    gradient[word] += d_loss_d_prob * d_angle

                # The bias enters only the decision angle, with coefficient 1.
                bias_gradient += d_loss_d_prob * d_decision

            scale = 1.0 / len(diagrams)
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


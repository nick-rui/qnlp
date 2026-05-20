"""A constrained grammar layer for the tutorial sentences.

The design document targets lambeq/CCG diagrams. For a classroom demo, the
dataset is intentionally small enough that we can make the grammar transparent
and deterministic while preserving the DisCoCat idea: grammatical reductions
decide how word meanings compose before the circuit is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrammarDiagram:
    """Parsed representation of a constrained tutorial sentence."""

    sentence: str
    tokens: tuple[str, ...]
    subject: str
    verb: str
    adjective: str | None = None
    object_: str | None = None
    object_adjective: str | None = None

    @property
    def has_object(self) -> bool:
        return self.object_ is not None

    @property
    def content_words(self) -> tuple[str, ...]:
        words: list[str] = []
        if self.adjective:
            words.append(self.adjective)
        words.append(self.subject)
        words.append(self.verb)
        if self.object_adjective:
            words.append(self.object_adjective)
        if self.object_:
            words.append(self.object_)
        return tuple(words)

    @property
    def roles(self) -> dict[str, str]:
        roles = {"subject": self.subject, "verb": self.verb}
        if self.adjective:
            roles["subject_modifier"] = self.adjective
        if self.object_:
            roles["object"] = self.object_
        if self.object_adjective:
            roles["object_modifier"] = self.object_adjective
        return roles

    def ascii_tree(self) -> str:
        """Return a compact syntax tree suitable for notebook display."""
        subject = f"Adj({self.adjective}) -> N({self.subject})" if self.adjective else f"N({self.subject})"
        if self.object_:
            if self.object_adjective:
                obj = f"Adj({self.object_adjective}) -> N({self.object_})"
            else:
                obj = f"N({self.object_})"
            return f"S\n|-- NP: {subject}\n`-- VP: V({self.verb}) -> NP({obj})"
        return f"S\n|-- NP: {subject}\n`-- VP: V({self.verb})"

    def string_diagram_text(self) -> str:
        """Describe the grammatical reductions in DisCoCat-style language."""
        lines = [
            f"sentence: {self.sentence}",
            "types: n^l n s for an intransitive verb, or n^l s n^r for a transitive verb",
            f"noun wire: {self.subject}",
        ]
        if self.adjective:
            lines.append(f"adjective box: {self.adjective} modifies {self.subject}")
        lines.append(f"verb box: {self.verb}")
        if self.object_:
            lines.append(f"object noun wire: {self.object_}")
            if self.object_adjective:
                lines.append(f"object adjective box: {self.object_adjective} modifies {self.object_}")
        lines.append("cups contract matching noun wires, leaving one sentence wire")
        return "\n".join(lines)


def parse_sentence(sentence: str) -> GrammarDiagram:
    """Parse one of the controlled tutorial sentence templates."""
    tokens = tuple(sentence.lower().strip().split())
    if not tokens or tokens[0] != "the":
        raise ValueError(f"Expected sentence to start with 'the': {sentence!r}")

    # Supported forms:
    #   the ADJ NOUN VERB
    #   the ADJ NOUN VERB the NOUN
    #   the ADJ NOUN VERB the ADJ NOUN
    if len(tokens) == 4:
        _, adjective, subject, verb = tokens
        return GrammarDiagram(
            sentence=sentence,
            tokens=tokens,
            adjective=adjective,
            subject=subject,
            verb=verb,
        )

    if len(tokens) == 6 and tokens[4] == "the":
        _, adjective, subject, verb, _, object_ = tokens
        return GrammarDiagram(
            sentence=sentence,
            tokens=tokens,
            adjective=adjective,
            subject=subject,
            verb=verb,
            object_=object_,
        )

    if len(tokens) == 7 and tokens[4] == "the":
        _, adjective, subject, verb, _, object_adjective, object_ = tokens
        return GrammarDiagram(
            sentence=sentence,
            tokens=tokens,
            adjective=adjective,
            subject=subject,
            verb=verb,
            object_=object_,
            object_adjective=object_adjective,
        )

    raise ValueError(f"Unsupported tutorial grammar template: {sentence!r}")


def parse_sentences(sentences: list[str] | tuple[str, ...]) -> list[GrammarDiagram]:
    """Parse a batch of tutorial sentences."""
    return [parse_sentence(sentence) for sentence in sentences]


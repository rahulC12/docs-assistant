"""Core data structures.

A document flows through the system as:

    Document  →  Page  →  Chunk  →  (indexed)  →  Hit  →  Answer

Page numbers are carried the whole way. They are the reason this tool
is useful rather than merely impressive: an answer you cannot trace
back to a page is an answer nobody can act on.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class Page:
    """One page of extracted text."""

    number: int
    text: str


@dataclass(slots=True)
class Document:
    """An uploaded file after text extraction."""

    doc_id: str
    filename: str
    pages: list[Page]
    added_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)

    @staticmethod
    def make_id(filename: str, content: bytes) -> str:
        """Content-addressed id, so re-uploading the same file is a no-op."""
        digest = hashlib.sha256(content).hexdigest()[:12]
        return f"{digest}"


@dataclass(slots=True)
class Chunk:
    """A retrievable passage.

    Chunks deliberately overlap. A sentence that answers the question
    is often split across a boundary, and overlap is the cheapest
    insurance against losing it.
    """

    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    position: int  # ordinal within the document

    @property
    def preview(self) -> str:
        flat = " ".join(self.text.split())
        return flat[:220] + ("…" if len(flat) > 220 else "")


@dataclass(slots=True)
class Hit:
    """A chunk retrieved for a query, with its score."""

    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class Citation:
    """A source the answer actually drew on."""

    marker: int  # the [1] shown inline in the answer
    filename: str
    page: int
    quote: str
    chunk_id: str


@dataclass
class Answer:
    """The response to a question."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: str = "medium"  # high | medium | low | none
    grounded: bool = True       # False when nothing relevant was found
    elapsed_ms: int = 0

    @property
    def is_refusal(self) -> bool:
        return not self.grounded

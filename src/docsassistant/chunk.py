"""Split documents into overlapping, retrievable passages.

Chunking is the least glamorous part of a RAG system and the part that
most determines whether it works. Two rules drive the implementation:

1. **Never split mid-sentence if avoidable.** A chunk that begins
   "...and shall not exceed 30 days" is useless as a citation, because
   a reader cannot tell what "shall not exceed" refers to.

2. **Overlap consecutive chunks.** The sentence that answers the
   question lands on a boundary more often than you would expect.
   Overlap is cheap insurance; the cost is a little duplication.
"""

from __future__ import annotations

import re

from .models import Chunk, Document

# Split on sentence-ending punctuation followed by whitespace and a
# capital or digit. Not linguistically perfect — deliberately so, since
# a heavier NLP dependency is not worth it for this job.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

# A heading is a short line with no terminal punctuation: "Sick Leave",
# "3. Termination", "APPENDIX B". Detecting these matters more than it
# looks — see `_split_sections`.
_HEADING = re.compile(r"^[^.!?;:]{2,80}$")

DEFAULT_CHUNK_WORDS = 110
DEFAULT_OVERLAP_WORDS = 25


def is_heading(line: str) -> bool:
    """Does this line look like a section heading?"""
    line = line.strip()
    if not line or len(line.split()) > 12:
        return False
    if not _HEADING.match(line):
        return False
    # A line ending in a comma or conjunction is a wrapped sentence,
    # not a heading.
    return not line.rstrip().endswith((",", "and", "or", "the", "of"))


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split page text into (heading, body) sections.

    Section boundaries are the single biggest lever on retrieval quality
    in this system. Without them, a chunk can straddle two unrelated
    topics — the sick-leave rules ending up inside the carry-over chunk,
    for instance — and every query that matches either topic retrieves a
    passage that is half irrelevant.

    Each chunk also carries its heading into its text, so a query for
    "sick leave" matches the section about sick leave even when the body
    never repeats the phrase.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if is_heading(block):
            if body:
                sections.append((heading, "\n\n".join(body)))
                body = []
            heading = block
        else:
            body.append(block)

    if body:
        sections.append((heading, "\n\n".join(body)))
    return sections


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized units.

    Line breaks count as boundaries alongside full stops. Plenty of real
    documents — CVs, spec sheets, tables of settings, bullet lists —
    contain almost no punctuation, and splitting on `.` alone leaves
    them as one enormous run. That produces citations that quote a whole
    page, which is exactly the failure this is here to prevent.
    """
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = _SENTENCE_END.split(line) if _SENTENCE_END.search(line) else [line]
        units.extend(p.strip() for p in parts if p.strip())
    return units


def chunk_document(
    document: Document,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Turn a document into overlapping chunks.

    Chunks never span a page or a section boundary, so every citation
    points at exactly one place in one document. The cost is slightly
    smaller chunks at boundaries, which is a good trade for a tool whose
    whole purpose is verifiable sources.
    """
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[Chunk] = []
    position = 0

    for page in document.pages:
        for heading, body in split_sections(page.text):
            sentences = split_sentences(body)
            if not sentences:
                # A heading with no body still belongs to the document.
                if heading:
                    sentences = [heading]
                else:
                    continue

            current: list[str] = []
            current_words = 0

            for sentence in sentences:
                sentence_words = len(sentence.split())

                # A sentence longer than the budget becomes its own chunk
                # rather than being cut in half.
                if sentence_words >= chunk_words:
                    position = _emit(
                        chunks, current, heading, document, page.number, position
                    )
                    current, current_words = [], 0
                    position = _emit(
                        chunks, [sentence], heading, document, page.number, position
                    )
                    continue

                if current_words + sentence_words > chunk_words and current:
                    position = _emit(
                        chunks, current, heading, document, page.number, position
                    )
                    current, current_words = _tail(current, overlap_words)

                current.append(sentence)
                current_words += sentence_words

            position = _emit(
                chunks, current, heading, document, page.number, position
            )

    return chunks


def _emit(
    chunks: list[Chunk],
    buffer: list[str],
    heading: str,
    document: Document,
    page_number: int,
    position: int,
) -> int:
    """Append one chunk built from `buffer`, and return the next position.

    A plain function rather than a closure over the loop variables:
    a nested function capturing `heading` and `page` is redefined on
    every iteration and reads them by reference, which is the classic
    late-binding trap even when — as here — it happens to be called
    before the loop advances.
    """
    # Join with newlines, not spaces: the answer layer re-splits chunk
    # text to quote only the relevant lines, and collapsing the
    # boundaries here leaves it nothing to split on.
    text = "\n".join(buffer).strip()
    if not text:
        return position

    # Prefix the heading so the chunk is self-describing — both for
    # retrieval and for the person reading the citation.
    if heading and not text.startswith(heading):
        text = f"{heading}\n{text}"

    chunks.append(
        Chunk(
            chunk_id=f"{document.doc_id}:{position}",
            doc_id=document.doc_id,
            filename=document.filename,
            page=page_number,
            text=text,
            position=position,
        )
    )
    return position + 1


def _tail(sentences: list[str], overlap_words: int) -> tuple[list[str], int]:
    """Return the trailing sentences to carry into the next chunk."""
    tail: list[str] = []
    words = 0
    for sentence in reversed(sentences):
        sentence_words = len(sentence.split())
        if words + sentence_words > overlap_words:
            break
        tail.insert(0, sentence)
        words += sentence_words
    return tail, words

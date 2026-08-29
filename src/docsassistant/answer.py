"""Turn retrieved passages into an answer the reader can verify.

Three rules shape this module, and they are what separate a useful tool
from a confident liar:

1. **Answer only from the retrieved passages.** The model is told, and
   the prompt gives it nothing else to work from.

2. **Every claim carries a citation.** Markers like [1] map to a real
   chunk with a filename and page number.

3. **Refuse when the documents do not contain the answer.** A system
   that always produces something is worse than useless on the
   questions where it has nothing — because the reader cannot tell the
   difference between a good answer and a fluent guess.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Protocol

from .models import Answer, Citation, Hit

# Fraction of the query's IDF-weighted terms that must appear in the
# best passage before we are willing to answer at all.
#
# Tuned on the labelled set in evaluate.py, not picked arbitrarily: at
# 0.30 the system answers genuine paraphrases ("minimum length" against
# "at least 14 characters") while refusing questions the corpus cannot
# support ("parental leave" against a sick-leave passage).
#
# This is a real trade-off dial, and the right value depends on the
# corpus. Raise it and the system refuses valid questions; lower it and
# it starts answering from passages that merely share a common word.
# Override per request with the `coverage_floor` argument.
COVERAGE_FLOOR = 0.30

# A weak secondary guard. Coverage does the real work; this only
# catches degenerate cases such as a single very common query term.
RELEVANCE_FLOOR = 0.3


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


def build_llm(name: str) -> LLM | None:
    """Return a provider, or None. Never raises — no key means fallback."""
    try:
        if name == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicLLM()
        if name == "openai" and os.environ.get("OPENAI_API_KEY"):
            return OpenAILLM()
    except ImportError:
        return None
    return None


# --- prompt -----------------------------------------------------------

_PROMPT = """Answer the question using ONLY the numbered sources below.

Rules:
- Cite every factual claim with its source number, like this: [1]
- Use several sources when the answer draws on more than one.
- If the sources do not contain the answer, reply with exactly:
  INSUFFICIENT
  followed by one sentence saying what is missing.
- Do not use outside knowledge. Do not guess.
- Be direct. Two or three sentences is usually enough.

Sources:
{sources}

Question: {question}

Reply as JSON only, no markdown fences:
{{"answer": "...", "confidence": "high|medium|low"}}"""


def _format_sources(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] {hit.chunk.filename}, page {hit.chunk.page}\n"
            f"{hit.chunk.text}"
        )
    return "\n\n".join(blocks)


def _parse_response(raw: str) -> tuple[str, str]:
    """Extract answer text and confidence, tolerating stray formatting."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        payload = json.loads(text)
        return str(payload.get("answer", "")), str(payload.get("confidence", "medium"))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                return (
                    str(payload.get("answer", "")),
                    str(payload.get("confidence", "medium")),
                )
            except json.JSONDecodeError:
                pass
    # The model ignored the format but may still have answered usefully.
    return text, "low"


_MARKER = re.compile(r"\[(\d+)\]")


def _collect_citations(
    answer_text: str, hits: list[Hit], question: str = ""
) -> tuple[str, list[Citation]]:
    """Map [n] markers onto real sources, dropping any the model invented.

    Models occasionally cite [7] when only four sources were supplied.
    Rendering that as a citation would be worse than having none, so
    unknown markers are stripped from the text entirely.
    """
    used: dict[int, Hit] = {}
    for marker in _MARKER.findall(answer_text):
        index = int(marker)
        if 1 <= index <= len(hits):
            used[index] = hits[index - 1]

    # Remove markers that point at nothing.
    def keep_or_strip(match: re.Match) -> str:
        return match.group(0) if int(match.group(1)) in used else ""

    cleaned = _MARKER.sub(keep_or_strip, answer_text)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned).strip()

    citations = [
        Citation(
            marker=index,
            filename=hit.chunk.filename,
            page=hit.chunk.page,
            quote=select_sentences(question, hit.chunk.text, limit=2),
            chunk_id=hit.chunk.chunk_id,
        )
        for index, hit in sorted(used.items())
    ]
    return cleaned, citations


# --- extractive fallback ---------------------------------------------

def select_sentences(question: str, text: str, limit: int = 3) -> str:
    """Pick the sentences in `text` that actually answer `question`.

    Quoting a whole retrieved passage is the fastest way to make a
    citation useless: the reader has to do the finding themselves,
    which is the work the tool was supposed to save. Retrieval picks
    the right passage; this picks the right lines inside it.

    Sentences are scored by IDF-ish overlap with the question — rare
    query terms count for more than common ones — and returned in their
    original order so the quote still reads naturally.
    """
    from .chunk import split_sentences
    from .index import entity_tokens, tokenize

    sentences = [s for s in split_sentences(text) if s.strip()]
    if len(sentences) <= 1:
        return " ".join(text.split())

    query_terms = set(tokenize(question))
    if not query_terms:
        return " ".join(sentences[:limit])

    # A term appearing in most sentences tells us little about which one
    # to pick, so weight by how rare it is within this passage.
    freq: dict[str, int] = {}
    # Include entity labels, or a question about an "email" scores zero
    # against the line holding the address and we fall back to the
    # opening lines — a citation that omits the evidence.
    tokenised = [tokenize(s) + entity_tokens(s) for s in sentences]
    for tokens in tokenised:
        for term in set(tokens):
            freq[term] = freq.get(term, 0) + 1

    scored = []
    for i, (sentence, tokens) in enumerate(zip(sentences, tokenised, strict=True)):
        present = set(tokens)
        score = sum(
            1.0 / (1 + freq.get(term, 0))
            for term in query_terms & present
        )
        scored.append((score, i, sentence))

    # Only keep lines that actually matched. Padding out to `limit`
    # staples an unrelated line onto every answer, which is worse than
    # a short quote — the reader cannot tell which part is the evidence.
    matched = [row for row in sorted(scored, reverse=True) if row[0] > 0]
    if not matched:
        return " ".join(sentences[:limit])
    best = matched[:limit]

    # A matched line that is only a heading ("PROJECTS") labels the
    # answer rather than being it, so carry the lines beneath it.
    indices: list[int] = []
    for _, i, sentence in best:
        indices.append(i)
        if len(sentence.split()) <= 3:
            indices.extend(range(i + 1, min(i + 4, len(sentences))))

    ordered = sorted(dict.fromkeys(indices))[: limit + 3]
    return " ".join(" ".join(sentences[i].split()) for i in ordered)


def extractive_answer(question: str, hits: list[Hit]) -> Answer:
    """Answer without an LLM by quoting the best passage.

    Not as fluent as a generated answer, and deliberately presented as
    a quotation rather than as prose, so nobody mistakes it for one.
    """
    if not hits:
        return _no_answer(question)

    best = hits[0]
    citations = [
        Citation(
            marker=i,
            filename=hit.chunk.filename,
            page=hit.chunk.page,
            quote=select_sentences(question, hit.chunk.text, limit=2),
            chunk_id=hit.chunk.chunk_id,
        )
        for i, hit in enumerate(hits[:3], start=1)
    ]

    body = select_sentences(question, best.chunk.text, limit=3)
    if len(body) > 700:
        body = body[:700].rsplit(" ", 1)[0] + "…"

    return Answer(
        question=question,
        text=(
            f"“{body}”\n\n"
            f"— {best.chunk.filename}, page {best.chunk.page} [1]"
        ),
        citations=citations,
        confidence="low",
        grounded=True,
    )


def _no_answer(question: str) -> Answer:
    return Answer(
        question=question,
        text=(
            "That isn't covered by the documents you've uploaded. "
            "Try rephrasing, or add the document that would contain it."
        ),
        citations=[],
        confidence="none",
        grounded=False,
    )


# --- entry point ------------------------------------------------------

_OVERVIEW = re.compile(
    r"\b(what(?:'s| is| are)?\s+(?:this|it|the\s+\w+)?\s*(?:document|file|pdf|paper|report)?"
    r"\s*(?:about|say|says|cover|contain)"
    r"|summar(?:y|ise|ize)|overview|tl;?dr|main\s+points?|key\s+points?"
    r"|gist|in\s+short|what\s+does\s+(?:this|it)\s+say)\b",
    re.IGNORECASE,
)

_GREETING = re.compile(
    r"^\s*(hi|hey|hello|yo|hola|namaste|good\s+(?:morning|afternoon|evening)|thanks?|"
    r"thank\s+you|ok(?:ay)?|test(?:ing)?)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def is_overview_question(question: str) -> bool:
    """Is this asking what the documents are, rather than about a detail?"""
    return bool(_OVERVIEW.search(question))


def is_greeting(question: str) -> bool:
    return bool(_GREETING.match(question))


def grade_confidence(
    coverage: float | None, unknown_ratio: float | None
) -> str:
    """Grade how much to trust a grounded answer.

    Deliberately advisory rather than a gate. Measured on the bundled
    eval set, neither coverage nor the unknown-term ratio separates
    answerable from unanswerable questions on its own:

        "What is the password minimum length?"  coverage 0.33, unknown 0.67
                                                → ANSWERABLE (the document
                                                  says "passphrase")
        "What is the company dress code?"       coverage 0.33, unknown 0.67
                                                → genuinely absent

    Those are the same numbers with opposite correct answers, because
    lexical retrieval cannot tell a vocabulary mismatch from a missing
    topic. Refusing on either signal would reject real questions.
    So the low-confidence band exists to tell the user "check the quoted
    passage carefully" rather than to silently withhold an answer.

    Dense embeddings narrow this gap, which is why enabling a provider
    measurably improves refusal quality.
    """
    if coverage is None:
        return "medium"
    if coverage >= 0.7 and (unknown_ratio or 0) <= 0.25:
        return "high"
    if coverage >= 0.45:
        return "medium"
    return "low"


def answer_question(
    question: str,
    hits: list[Hit],
    llm: LLM | None = None,
    relevance_floor: float = RELEVANCE_FLOOR,
    coverage: float | None = None,
    coverage_floor: float = COVERAGE_FLOOR,
    unknown_ratio: float | None = None,
) -> Answer:
    """Produce an answer from retrieved passages.

    `coverage` is the IDF-weighted term coverage of the best hit, from
    Library.coverage(). When supplied it is the primary test of whether
    the documents can answer the question at all.
    """
    started = time.perf_counter()

    if not question.strip():
        return _no_answer(question)

    if is_greeting(question):
        result = Answer(
            question=question,
            text=(
                "Hello. Ask me something about the documents you've added — "
                "I'll answer from them and show you the page it came from."
            ),
            citations=[],
            confidence="none",
            grounded=True,
        )
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    # An overview question has no distinctive terms to rank or score by,
    # so the coverage gate would refuse it. The caller supplies the
    # documents' opening passages instead; trust them.
    overview = is_overview_question(question)
    if overview:
        coverage = None
        unknown_ratio = None

    # Coverage is the gate whenever it is available. A raw BM25 score
    # cannot be thresholded: it scales with corpus size and query
    # length, so on a small library every score sits below any fixed
    # floor even when the passage plainly answers the question. The
    # absolute floor is kept only as a fallback for callers that cannot
    # compute coverage.
    if not hits:
        insufficient = True
    elif overview:
        insufficient = False
    elif coverage is not None:
        insufficient = coverage < coverage_floor
    else:
        insufficient = hits[0].score < relevance_floor
    if insufficient:
        result = _no_answer(question)
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    if llm is None:
        result = extractive_answer(question, hits)
        result.confidence = grade_confidence(coverage, unknown_ratio)
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    try:
        raw = llm.complete(
            _PROMPT.format(sources=_format_sources(hits), question=question)
        )
        text, confidence = _parse_response(raw)
    except Exception as exc:  # degrade to extractive rather than fail
        result = extractive_answer(question, hits)
        result.text = f"{result.text}\n\n(AI provider unavailable: {exc})"
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    if text.strip().upper().startswith("INSUFFICIENT"):
        detail = text.split("\n", 1)[1].strip() if "\n" in text else ""
        result = _no_answer(question)
        if detail:
            result.text = detail
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    cleaned, citations = _collect_citations(text, hits, question)

    # An answer with no surviving citations is not grounded, whatever
    # the model claimed.
    grounded = bool(citations)

    result = Answer(
        question=question,
        text=cleaned,
        citations=citations,
        confidence=confidence if grounded else "low",
        grounded=grounded,
    )
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result
"""Retrieval: BM25 by default, dense embeddings when available.

Why BM25 is the default
-----------------------
Most RAG tutorials reach straight for embeddings, which means the
project cannot run without either an API key or a few hundred megabytes
of model download. That is a poor default for a tool someone should be
able to try in thirty seconds.

BM25 is a lexical ranking function that has been the information
retrieval baseline for thirty years. It needs no model, no API and no
GPU, and on document Q&A — where users tend to use the same vocabulary
as the documents — it is a genuinely strong retriever, not a
placeholder.

Dense embeddings are better at paraphrase ("holiday entitlement" vs
"annual leave"), so when they are available the two are combined with
Reciprocal Rank Fusion rather than one replacing the other.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

from .models import Chunk, Hit

_TOKEN = re.compile(r"[a-z0-9]+")



# Words too common to discriminate between passages. Kept deliberately
# short — over-aggressive stopword lists hurt phrase queries.
_STOPWORDS = frozenset([
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot", "could",
    "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "has", "have", "having", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "let", "me", "more", "most", "much", "must",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "out", "over", "own", "same", "shall",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours",
])


# Deliberately crude suffix stripping rather than a full Porter stemmer.
# The goal is to match "sickness" with "sick" and "requests" with
# "request"; aggressive stemming conflates words that mean different
# things ("policy"/"police") and hurts precision more than it helps.
_SUFFIXES = ("iness", "ness", "ingly", "edly", "ing", "ies", "ied", "es", "ed", "ly", "s")


def stem(token: str) -> str:
    """Strip one common suffix, if the remainder is still a real word length."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            # "ss" is not a plural. Without this guard the stemmer turns
            # dress → dres, address → addres, process → proces and
            # business → busines, so a query for any of them fails to
            # match the documents that contain them.
            if suffix == "s" and token.endswith("ss"):
                return token
            base = token[: -len(suffix)]
            # "ies" -> "y" reads better than a bare stem ("policies" -> "polic")
            if suffix in ("ies", "ied"):
                return base + "y"
            return base
    return token


_ENTITY_PATTERNS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
     ("email", "mail", "contact", "address")),
    (re.compile(r"(?:\+\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}|\(\d{3}\)\s?\d{3}-\d{4}"),
     ("phone", "mobile", "number", "contact")),
    (re.compile(r"https?://\S+|\b(?:www\.|github\.com/|linkedin\.com/)\S+"),
     ("link", "url", "website", "profile")),
    (re.compile(r"\b(?:19|20)\d{2}\b"), ("year", "date")),
]


def entity_tokens(text: str) -> list[str]:
    """Extra index terms for things people name but documents don't.

    A CV prints `you@example.com` but never the word "email", so a
    question about someone's email address matches nothing. The same
    applies to phone numbers, profile links and dates.

    These terms are added at INDEX time only, never to the query. That
    is document expansion, not query rewriting: it widens what a passage
    can be found by without changing what the user asked. Query-side
    expansion would be the mistake here, because it would make every
    question match everything.
    """
    extra: list[str] = []
    for pattern, labels in _ENTITY_PATTERNS:
        if pattern.search(text):
            extra.extend(labels)
    return [stem(t) for t in extra]


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, and stem.

    Both indexing and querying go through this function, so the two
    always agree. A mismatch between index-time and query-time
    processing is a classic silent retrieval bug.
    """
    return [
        stem(t)
        for t in _TOKEN.findall(text.lower())
        if t not in _STOPWORDS
    ]


class BM25Index:
    """Okapi BM25 over a chunk collection.

    Parameters follow the standard defaults from the literature:
    k1 controls term-frequency saturation (how quickly repeated terms
    stop adding score) and b controls length normalisation (how much a
    long passage is penalised for its length).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._term_freqs: list[Counter] = []
        self._lengths: list[int] = []
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._postings: dict[str, set[int]] = defaultdict(set)
        self._avg_length = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            # Entity labels are added at index time only — see entity_tokens.
            tokens = tokenize(chunk.text) + entity_tokens(chunk.text)
            if not tokens:
                continue
            index = len(self.chunks)
            self.chunks.append(chunk)
            freqs = Counter(tokens)
            self._term_freqs.append(freqs)
            self._lengths.append(len(tokens))
            for term in freqs:
                self._doc_freq[term] += 1
                self._postings[term].add(index)

        self._avg_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    def remove_document(self, doc_id: str) -> int:
        """Drop a document and rebuild. Rebuilding is simpler than
        maintaining deletion tombstones, and at this scale it is fast."""
        keep = [c for c in self.chunks if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.__init__(k1=self.k1, b=self.b)
            self.add(keep)
        return removed

    def _idf(self, term: str) -> float:
        """Inverse document frequency, floored at zero.

        The +0.5 smoothing can go negative for terms appearing in more
        than half the collection; clamping avoids a term subtracting
        from a passage's score.
        """
        n = len(self.chunks)
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return max(0.0, math.log(1 + (n - df + 0.5) / (df + 0.5)))

    def search(self, query: str, top_k: int = 8) -> list[Hit]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []

        # Only score passages containing at least one query term.
        candidates: set[int] = set()
        for term in terms:
            candidates |= self._postings.get(term, set())
        if not candidates:
            return []

        scores: dict[int, float] = {}
        for index in candidates:
            freqs = self._term_freqs[index]
            length = self._lengths[index]
            score = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                norm = 1 - self.b + self.b * (length / (self._avg_length or 1))
                score += self._idf(term) * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * norm
                )
            if score > 0:
                scores[index] = score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            Hit(chunk=self.chunks[i], score=round(s, 4), rank=rank)
            for rank, (i, s) in enumerate(ranked, start=1)
        ]

    def _max_idf(self) -> float:
        """IDF of a term appearing in exactly one chunk."""
        n = len(self.chunks)
        if n == 0:
            return 1.0
        return math.log(1 + (n - 1 + 0.5) / 1.5)

    def unknown_term_ratio(self, query: str) -> float:
        """Fraction of the query's content terms the collection has never seen.

        This is a sharper signal than coverage, and a different one.
        Coverage asks whether the *best passage* contains the query's
        terms; this asks whether the *entire collection* does.

        If someone asks about a dress code and the word "dress" appears
        in none of the indexed documents, no amount of ranking will
        produce a real answer — the topic simply is not in the corpus.
        Coverage alone under-weights this, because an unseen term can
        only be scored using the collection's own IDF range, which on a
        small corpus is narrow. Treating it as a separate test avoids
        tuning one threshold to do two different jobs.
        """
        terms = set(tokenize(query))
        if not terms:
            return 1.0
        unknown = sum(1 for term in terms if term not in self._doc_freq)
        return unknown / len(terms)

    def coverage(self, query: str, chunk: Chunk) -> float:
        """How much of the query's *meaning* this chunk actually contains.

        A raw BM25 score cannot answer "is this good enough?", because
        scores are not comparable across queries — a long query or a
        rare term inflates them regardless of relevance. Coverage can:
        it asks what fraction of the query's distinctive terms appear
        in the passage at all.

        Terms are weighted by IDF, so matching "parental" counts for far
        more than matching "leave". Without that weighting, a question
        about parental leave scores well against any passage mentioning
        leave, which is exactly the failure this is here to prevent.
        """
        terms = set(tokenize(query))
        if not terms:
            return 0.0

        # Match what add() indexed, or a passage found via an entity
        # label scores zero coverage and gets refused.
        present = set(tokenize(chunk.text)) | set(entity_tokens(chunk.text))
        total_weight = 0.0
        matched_weight = 0.0

        for term in terms:
            # An unknown term is maximally distinctive — if the
            # collection has never seen "parental", a passage lacking it
            # is a poor match. It gets the weight a term appearing in
            # exactly one chunk would get, rather than an arbitrary
            # large constant, so the scale stays tied to this collection.
            weight = (
                self._idf(term) if term in self._doc_freq else self._max_idf()
            )
            total_weight += weight
            if term in present:
                matched_weight += weight

        if total_weight == 0:
            return 0.0
        return matched_weight / total_weight

    def __len__(self) -> int:
        return len(self.chunks)


# --- optional dense retrieval ----------------------------------------

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Dense embeddings via the OpenAI API. Optional."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class DenseIndex:
    """Cosine-similarity search over embedding vectors."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch])
            self.chunks.extend(batch)
            self._vectors.extend(_normalise(v) for v in vectors)

    def search(self, query: str, top_k: int = 8) -> list[Hit]:
        if not self._vectors:
            return []
        q = _normalise(self.embedder.embed([query])[0])
        scored = [
            (i, sum(a * b for a, b in zip(q, v, strict=True)))
            for i, v in enumerate(self._vectors)
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [
            Hit(chunk=self.chunks[i], score=round(s, 4), rank=rank)
            for rank, (i, s) in enumerate(scored[:top_k], start=1)
        ]


def _normalise(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / magnitude for x in vector]


# --- fusion ------------------------------------------------------------

def reciprocal_rank_fusion(
    result_sets: list[list[Hit]], k: int = 60, top_k: int = 8
) -> list[Hit]:
    """Combine several ranked lists into one.

    RRF scores each item by 1/(k + rank) summed across the lists it
    appears in. It needs no score calibration between retrievers, which
    matters here because BM25 scores and cosine similarities are not on
    remotely the same scale. The constant k damps the influence of very
    high ranks; 60 is the value from the original paper.
    """
    totals: dict[str, float] = defaultdict(float)
    by_id: dict[str, Chunk] = {}

    for hits in result_sets:
        for hit in hits:
            totals[hit.chunk.chunk_id] += 1.0 / (k + hit.rank)
            by_id[hit.chunk.chunk_id] = hit.chunk

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        Hit(chunk=by_id[cid], score=round(score, 6), rank=rank)
        for rank, (cid, score) in enumerate(ranked, start=1)
    ]


# --- the store the app actually uses ----------------------------------

class Library:
    """Everything indexed, plus persistence.

    Chunks are stored as JSON rather than in a vector database. At the
    scale this tool targets — one person's document set — a database
    would be infrastructure without benefit, and a readable file is far
    easier to inspect when something looks wrong.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.bm25 = BM25Index()
        self.dense = DenseIndex(embedder) if embedder else None
        self.documents: dict[str, dict] = {}

    @property
    def is_hybrid(self) -> bool:
        return self.dense is not None

    def add_document(self, doc_id: str, filename: str,
                     chunks: list[Chunk], pages: int) -> None:
        self.bm25.add(chunks)
        if self.dense is not None:
            self.dense.add(chunks)
        self.documents[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "pages": pages,
            "chunks": len(chunks),
        }

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        self.bm25.remove_document(doc_id)
        del self.documents[doc_id]
        return True

    def _max_idf(self) -> float:
        """IDF of a term appearing in exactly one chunk."""
        n = len(self.chunks)
        if n == 0:
            return 1.0
        return math.log(1 + (n - 1 + 0.5) / 1.5)

    def coverage(self, query: str, chunk: Chunk) -> float:
        return self.bm25.coverage(query, chunk)

    def unknown_term_ratio(self, query: str) -> float:
        return self.bm25.unknown_term_ratio(query)

    def leading(self, limit: int = 5) -> list[Hit]:
        """The opening passages of each document, in order.

        Overview questions — "what is this about", "summarise this" —
        contain no distinctive terms to rank by, so BM25 has nothing to
        work with and coverage is necessarily zero. Refusing them is
        wrong: the documents plainly *can* answer, and it is the most
        natural first question anyone asks.

        Documents lead with what they are about, so their opening
        passages are the right context for a summary.
        """
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in self.bm25.chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)

        picked: list[Chunk] = []
        per_doc = max(1, limit // max(1, len(by_doc)))
        for chunks in by_doc.values():
            ordered = sorted(chunks, key=lambda c: c.position)
            picked.extend(ordered[:per_doc])

        return [Hit(chunk=c, score=1.0) for c in picked[:limit]]

    def search(self, query: str, top_k: int = 6) -> list[Hit]:
        lexical = self.bm25.search(query, top_k=top_k * 2)
        if self.dense is None:
            return lexical[:top_k]
        semantic = self.dense.search(query, top_k=top_k * 2)
        return reciprocal_rank_fusion([lexical, semantic], top_k=top_k)

    @property
    def chunk_count(self) -> int:
        return len(self.bm25)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self.documents,
            "chunks": [
                {
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                    "filename": c.filename, "page": c.page,
                    "text": c.text, "position": c.position,
                }
                for c in self.bm25.chunks
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedder: Embedder | None = None) -> Library:
        library = cls(embedder=embedder)
        if not path.exists():
            return library
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in payload.get("chunks", [])]
        library.bm25.add(chunks)
        if library.dense is not None and chunks:
            library.dense.add(chunks)
        library.documents = payload.get("documents", {})
        return library

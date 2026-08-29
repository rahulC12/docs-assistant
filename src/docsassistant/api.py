"""HTTP API and static file serving.

Upload is synchronous because extraction and indexing of a normal
document takes well under a second — a job-polling flow would be
ceremony without benefit here. If you index very large PDFs, move
`_ingest` onto a background task and poll, exactly as the log analyzer
does.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer import answer_question, build_llm, is_overview_question
from .chunk import chunk_document
from .extract import ExtractionError, extract
from .index import Library

WEB_DIR = Path(__file__).parent / "web"
DATA_FILE = Path(os.environ.get("DATA_FILE", ".data/library.json"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
PROVIDER = os.environ.get("LLM_PROVIDER", "none")

app = FastAPI(
    title="Docs Assistant",
    description="Ask questions of your documents and get answers with sources.",
    version="0.1.0",
)

library: Library = Library.load(DATA_FILE)


# --- schemas ----------------------------------------------------------

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


class LibraryStatus(BaseModel):
    documents: list[DocumentInfo]
    chunk_count: int
    retriever: str
    provider: str
    answers_with_ai: bool


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=12)


class CitationOut(BaseModel):
    marker: int
    filename: str
    page: int
    quote: str
    chunk_id: str


class AnswerOut(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    confidence: str
    grounded: bool
    elapsed_ms: int


# --- helpers ----------------------------------------------------------

def _status() -> LibraryStatus:
    return LibraryStatus(
        documents=[DocumentInfo(**d) for d in library.documents.values()],
        chunk_count=library.chunk_count,
        retriever="hybrid (BM25 + embeddings)" if library.is_hybrid else "BM25",
        provider=PROVIDER,
        answers_with_ai=build_llm(PROVIDER) is not None,
    )


# --- routes -----------------------------------------------------------

@app.get("/api/status", response_model=LibraryStatus)
def status() -> LibraryStatus:
    return _status()


@app.post("/api/documents", response_model=LibraryStatus)
async def upload(files: list[UploadFile] = File(...)) -> LibraryStatus:
    if not files:
        raise HTTPException(400, "No files were uploaded.")

    errors: list[str] = []

    for upload_file in files:
        content = await upload_file.read()

        if len(content) > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            errors.append(f"{upload_file.filename} is larger than {limit_mb}MB.")
            continue

        try:
            document = extract(upload_file.filename or "untitled", content)
        except ExtractionError as exc:
            errors.append(f"{upload_file.filename}: {exc}")
            continue

        if document.doc_id in library.documents:
            continue  # same content already indexed

        chunks = chunk_document(document)
        library.add_document(
            document.doc_id, document.filename, chunks, document.page_count
        )

    library.save(DATA_FILE)

    if errors and not library.documents:
        raise HTTPException(400, " ".join(errors))
    if errors:
        # Partial success: report what failed without discarding what worked.
        raise HTTPException(207, " ".join(errors))

    return _status()


@app.delete("/api/documents/{doc_id}", response_model=LibraryStatus)
def remove(doc_id: str) -> LibraryStatus:
    if not library.remove_document(doc_id):
        raise HTTPException(404, "No document with that id.")
    library.save(DATA_FILE)
    return _status()


@app.post("/api/ask", response_model=AnswerOut)
def ask(request: AskRequest) -> AnswerOut:
    if not library.documents:
        raise HTTPException(400, "Add a document before asking a question.")

    if is_overview_question(request.question):
        # Nothing to rank by — hand over the documents' opening passages.
        hits = library.leading(limit=request.top_k)
        coverage = None
    else:
        hits = library.search(request.question, top_k=request.top_k)
        coverage = library.coverage(request.question, hits[0].chunk) if hits else 0.0
    result = answer_question(
        request.question,
        hits,
        llm=build_llm(PROVIDER),
        coverage=coverage,
        unknown_ratio=library.unknown_term_ratio(request.question),
    )

    return AnswerOut(
        question=result.question,
        answer=result.text,
        citations=[CitationOut(**vars(c)) for c in result.citations],
        confidence=result.confidence,
        grounded=result.grounded,
        elapsed_ms=result.elapsed_ms,
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "documents": len(library.documents)}


# --- static -----------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
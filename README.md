# Docs Assistant

Ask questions about your own documents and get an answer **with the page it came from**.

![tests](https://img.shields.io/badge/tests-50%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![no api key required](https://img.shields.io/badge/API%20key-optional-informational)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The problem

People don't distrust AI answers because they're badly written. They distrust them because there is no way to check them.

A contract review tool that says *"termination requires 90 days notice"* is useless if you then have to read the contract yourself to find out whether that's true. The answer is only worth something when it comes with the receipt.

So this is built around the citation rather than the chat. Every answer names the file, the page, and quotes the passage it came from. When your documents don't cover the question, it says so instead of inventing something.

## What it does

- **Upload** PDF, Word, text or markdown files
- **Ask** questions in plain language
- **Get answers with page-level citations** — click a `[1]` in the answer to jump to the passage it used
- **Refuses when it should** — no sources, no answer
- **Grades its own confidence** so you know when to double-check
- **Runs with no API key** — BM25 retrieval and quoted passages out of the box

## Quick start

```bash
git clone https://github.com/rahulC12/docs-assistant
cd docs-assistant
pip install -r requirements-dev.txt
pip install -e .

uvicorn docsassistant.api:app --reload
```

Open **http://localhost:8000**, drop in a document, ask a question.

Or with Docker:

```bash
docker compose up
```

### From the terminal

```bash
docs-assistant add handbook.pdf policy.docx
docs-assistant ask "How many days of annual leave do employees get?"
docs-assistant list
```

## How it works

```
document
   │
   ├─ 1. extract    PDF / DOCX / TXT / MD → text, page numbers kept
   │
   ├─ 2. chunk      split on structure, overlap at the seams
   │                every chunk remembers its file and page
   │
   ├─ 3. index      BM25 by default; dense embeddings if available
   │
   ├─ 4. retrieve   rank passages, fuse rankings if both retrievers run
   │
   ├─ 5. ground     is this good enough to answer at all?
   │                if not, refuse — this step is the product
   │
   └─ 6. answer     LLM writes prose from the passages, or the best
                    passage is quoted directly when no key is set
```

### Why BM25 rather than embeddings by default

Most RAG projects reach straight for embeddings, which means they cannot run without either an API key or a few hundred megabytes of model download. That's a poor default for something you should be able to try in thirty seconds.

BM25 has been the information-retrieval baseline for thirty years. It needs no model, no API and no GPU, and on document Q&A — where people tend to use the same words as their documents — it's a genuinely strong retriever rather than a placeholder.

Embeddings are better at paraphrase (*"holiday entitlement"* vs *"annual leave"*), so when they're available the two rankings are combined with Reciprocal Rank Fusion rather than one replacing the other.

```bash
pip install sentence-transformers   # dense retrieval switches on automatically
```

### Why refusing is the hard part

Retrieval always returns *something* — the ranking is relative, so the best of a bad set still comes back ranked first. Deciding whether that best match is good enough is a separate problem, and it's the one that decides whether the tool can be trusted.

Two signals are computed for every question:

- **Coverage** — what fraction of the question's distinctive terms appear in the retrieved passage, weighted by IDF so matching *"parental"* counts for far more than matching *"leave"*.
- **Unknown-term ratio** — what fraction of the question's terms appear nowhere in the entire collection.

Coverage below the floor means refuse. That threshold is calibrated against the bundled evaluation set, not guessed.

## Results

Run `python -m docsassistant.evaluate` to reproduce these.

| Metric | Result |
|---|---|
| Recall@1 | 1.00 |
| Recall@3 | 1.00 |
| MRR | 1.00 |
| Correct refusals | 2/2 |
| False refusals | 0 |
| Answered when it shouldn't have | 0 |
| Tests | 50 passing, 83% coverage |

**Read the next section before trusting those numbers.**

## Limitations

These are measured, not guessed. A tool like this is only useful if you know where it stops working.

**The evaluation set is small — 14 questions over 3 synthetic documents.** Perfect scores on it mean the system isn't obviously broken, not that it's accurate in general. Treat it as a regression test, not a benchmark.

**Refusal is unreliable in the middle band, and this is fundamental rather than a tuning problem.** Measured on the bundled corpus:

| Question | Coverage | Unknown terms | Truth |
|---|---|---|---|
| "What is the password minimum length?" | 0.33 | 0.67 | **answerable** — the document says *passphrase* |
| "What is the company dress code?" | 0.33 | 0.67 | **not in the documents** |

Identical signals, opposite correct answers. Lexical retrieval cannot distinguish a vocabulary mismatch from a missing topic, so any threshold that catches the second rejects the first. Rather than pick which error to make silently, borderline answers are returned marked **low confidence** with the passage quoted, so you can judge in one glance. Installing `sentence-transformers` measurably narrows this gap.

**Answers without an LLM are quoted lines, not prose.** Retrieval finds the passage and the relevant lines inside it, but nothing rewrites them into a sentence. Set `LLM_PROVIDER` for written answers.

**Other known gaps:**

- **Tables and multi-column PDFs extract poorly.** `pypdf` reads them as jumbled text, so a question answered by a table often fails. Scanned PDFs return nothing at all — there is no OCR.
- **No cross-document reasoning.** "Which of these three contracts has the longest notice period?" needs comparison across passages; retrieval returns passages independently.
- **The index is in memory, persisted as JSON.** Fine to tens of thousands of passages, wrong for millions — that needs a real vector database.
- **Single process.** The library is a module-level object, so running more than one worker gives each its own copy.
- **No authentication.** Anyone who can reach the port can read every uploaded document. Do not put this on the open internet as-is.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `none` | `anthropic`, `openai`, or `none` for quoted passages |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | only needed for written answers |
| `DATA_FILE` | `.data/library.json` | where the index is persisted |
| `MAX_UPLOAD_BYTES` | 25 MB | upload limit |

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | what's indexed, which retriever is active |
| `POST /api/documents` | upload and index files |
| `DELETE /api/documents/{id}` | remove a document |
| `POST /api/ask` | ask a question, get answer + citations |
| `GET /api/health` | liveness |

Interactive docs at `/docs` once running.

## Development

```bash
pip install -r requirements-dev.txt
pytest --cov=docsassistant --cov-report=term-missing
ruff check src tests
python -m docsassistant.evaluate     # retrieval quality gate
```

The frontend is three static files — `index.html`, `style.css`, `app.js` — served directly by FastAPI. No build step, no `node_modules`.

## Tech stack

Python 3.10+ · FastAPI · BM25 (implemented from scratch) · pypdf · python-docx · Docker · GitHub Actions · pytest · ruff

## License

MIT
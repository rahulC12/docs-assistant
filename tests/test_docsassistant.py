"""Tests for extraction, chunking, retrieval, grounding and the API."""

from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")

from docsassistant.answer import grade_confidence  # noqa: E402
from docsassistant.chunk import chunk_document  # noqa: E402
from docsassistant.extract import ExtractionError, extract  # noqa: E402
from docsassistant.index import Library, stem, tokenize  # noqa: E402

# --- tokenisation -----------------------------------------------------

class TestTokenizer:
    @pytest.mark.parametrize("word,expected", [
        ("dress", "dress"),
        ("address", "address"),
        ("process", "process"),
        ("class", "class"),
        ("policies", "policy"),
        ("leave", "leave"),
    ])
    def test_double_s_is_not_a_plural(self, word, expected):
        """Regression: the stemmer turned dress → dres, address → addres,
        so queries for those words matched nothing."""
        assert stem(word) == expected

    def test_stopwords_removed(self):
        assert "the" not in tokenize("the leave policy")

    def test_index_and_query_agree(self):
        """Index-time and query-time processing must match exactly, or
        retrieval silently fails."""
        assert tokenize("Annual Leave Policies") == tokenize("annual leave policies")


# --- extraction -------------------------------------------------------

class TestExtraction:
    def test_plain_text(self):
        doc = extract("notes.txt", b"line one\nline two")
        assert doc.page_count == 1
        assert "line one" in doc.pages[0].text

    def test_markdown(self):
        doc = extract("readme.md", b"# Title\n\nBody text here.")
        assert "Body text here." in doc.pages[0].text

    def test_unsupported_type_rejected(self):
        with pytest.raises(ExtractionError):
            extract("photo.png", b"\x89PNG\r\n")

    def test_empty_file_rejected(self):
        with pytest.raises(ExtractionError):
            extract("empty.txt", b"   \n  ")

    def test_pdf_pages_are_numbered(self, tmp_path):
        pytest.importorskip("reportlab")
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        path = tmp_path / "two.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        c.drawString(80, 700, "First page about annual leave")
        c.showPage()
        c.drawString(80, 700, "Second page about expenses")
        c.showPage()
        c.save()

        doc = extract("two.pdf", path.read_bytes())
        assert doc.page_count == 2
        assert doc.pages[0].number == 1
        assert "annual leave" in doc.pages[0].text
        assert "expenses" in doc.pages[1].text


# --- chunking ---------------------------------------------------------

class TestChunking:
    def test_page_number_survives_chunking(self):
        doc = extract("policy.txt", b"Leave rules here.")
        chunks = chunk_document(doc)
        assert all(c.page >= 1 for c in chunks)
        assert all(c.filename == "policy.txt" for c in chunks)

    def test_long_text_splits(self):
        body = ("Sentence number one is here. " * 200).encode()
        chunks = chunk_document(extract("long.txt", body))
        assert len(chunks) > 1

    def test_chunks_have_unique_ids(self):
        body = ("Some content to split up. " * 200).encode()
        chunks = chunk_document(extract("long.txt", body))
        assert len({c.chunk_id for c in chunks}) == len(chunks)


# --- retrieval --------------------------------------------------------

@pytest.fixture
def library():
    lib = Library()
    doc = extract(
        "handbook.txt",
        b"Annual Leave\n\nEmployees receive 25 days of paid annual leave "
        b"each year.\n\nExpenses\n\nThe daily meal allowance is 30 pounds.",
    )
    lib.add_document(doc.doc_id, doc.filename, chunk_document(doc), doc.page_count)
    return lib


class TestRetrieval:
    def test_finds_the_right_passage(self, library):
        hits = library.search("How much meal allowance?", top_k=3)
        assert "meal allowance" in hits[0].chunk.text

    def test_coverage_high_for_on_topic(self, library):
        hits = library.search("annual leave days", top_k=1)
        assert library.coverage("annual leave days", hits[0].chunk) > 0.5

    def test_unknown_terms_detected(self, library):
        assert library.unknown_term_ratio("dress code policy") > 0.5
        assert library.unknown_term_ratio("annual leave") == 0.0

    def test_removing_a_document_empties_the_index(self, library):
        doc_id = next(iter(library.documents))
        assert library.remove_document(doc_id) is True
        assert library.search("anything") == []

    def test_removing_unknown_document_is_false(self, library):
        assert library.remove_document("nope") is False


# --- confidence -------------------------------------------------------

class TestConfidence:
    def test_strong_match_is_high(self):
        assert grade_confidence(0.8, 0.0) == "high"

    def test_weak_match_is_low(self):
        assert grade_confidence(0.33, 0.67) == "low"

    def test_partial_match_is_medium(self):
        assert grade_confidence(0.5, 0.4) == "medium"

    def test_missing_signal_defaults_to_medium(self):
        assert grade_confidence(None, None) == "medium"


# --- API --------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_FILE", str(tmp_path / "lib.json"))
    import importlib

    from docsassistant import api as api_module
    importlib.reload(api_module)
    return TestClient(api_module.app)


DOC = (
    b"Annual Leave\n\nEmployees receive 25 days of paid annual leave per year. "
    b"Requests need 14 days notice.\n\nExpenses\n\nThe meal allowance is 30 pounds "
    b"per day and claims close after 60 days."
)


class TestAPI:
    def test_health(self, client):
        assert client.get("/api/health").json()["status"] == "ok"

    def test_serves_the_interface(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Docs" in response.text

    def test_empty_library_status(self, client):
        body = client.get("/api/status").json()
        assert body["documents"] == []
        assert body["chunk_count"] == 0

    def test_upload_then_ask_with_citation(self, client):
        upload = client.post(
            "/api/documents", files=[("files", ("handbook.txt", DOC, "text/plain"))]
        )
        assert upload.status_code == 200
        assert upload.json()["documents"][0]["filename"] == "handbook.txt"

        answer = client.post(
            "/api/ask", json={"question": "How many days of annual leave?"}
        ).json()

        assert answer["grounded"] is True
        assert answer["citations"], "a grounded answer must cite something"
        assert answer["citations"][0]["page"] >= 1
        assert "25 days" in answer["citations"][0]["quote"]

    def test_refuses_when_topic_is_absent(self, client):
        client.post(
            "/api/documents", files=[("files", ("handbook.txt", DOC, "text/plain"))]
        )
        answer = client.post(
            "/api/ask", json={"question": "Who won the 1998 world cup final?"}
        ).json()

        assert answer["grounded"] is False
        assert answer["citations"] == []

    def test_ask_before_upload_is_rejected(self, client):
        assert client.post("/api/ask", json={"question": "anything"}).status_code == 400

    def test_blank_question_rejected(self, client):
        client.post(
            "/api/documents", files=[("files", ("h.txt", DOC, "text/plain"))]
        )
        assert client.post("/api/ask", json={"question": ""}).status_code == 422

    def test_unsupported_file_rejected(self, client):
        response = client.post(
            "/api/documents", files=[("files", ("image.png", b"\x89PNG", "image/png"))]
        )
        assert response.status_code == 400

    def test_delete_document(self, client):
        upload = client.post(
            "/api/documents", files=[("files", ("h.txt", DOC, "text/plain"))]
        ).json()
        doc_id = upload["documents"][0]["doc_id"]

        after = client.delete(f"/api/documents/{doc_id}").json()
        assert after["documents"] == []

    def test_delete_unknown_document_404(self, client):
        assert client.delete("/api/documents/nope").status_code == 404

    def test_library_persists_across_restart(self, client, tmp_path, monkeypatch):
        client.post(
            "/api/documents", files=[("files", ("h.txt", DOC, "text/plain"))]
        )
        import importlib

        from docsassistant import api as api_module
        importlib.reload(api_module)
        reloaded = TestClient(api_module.app)
        assert len(reloaded.get("/api/status").json()["documents"]) == 1


# --- retrieval quality ------------------------------------------------

class TestEvaluation:
    def test_eval_suite_meets_its_bar(self):
        """The published quality numbers must stay true."""
        from docsassistant.evaluate import evaluate

        report = evaluate()
        assert report["recall_at_3"] == 1.0
        assert report["answered_when_it_should_not"] == 0
        assert report["false_refusals"] == 0


# --- CLI --------------------------------------------------------------

class TestCLI:
    @pytest.fixture(autouse=True)
    def _isolated_library(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_FILE", str(tmp_path / "lib.json"))
        import importlib

        from docsassistant import cli as cli_module
        importlib.reload(cli_module)
        self.cli = cli_module

    def _doc(self, tmp_path):
        path = tmp_path / "handbook.txt"
        path.write_bytes(DOC)
        return path

    def test_add_then_list(self, tmp_path, capsys):
        assert self.cli.main(["add", str(self._doc(tmp_path))]) == 0
        assert self.cli.main(["list"]) == 0
        assert "handbook.txt" in capsys.readouterr().out

    def test_add_missing_file_returns_error(self, tmp_path, capsys):
        assert self.cli.main(["add", str(tmp_path / "nope.txt")]) == 1

    def test_add_unsupported_type_is_skipped(self, tmp_path):
        bad = tmp_path / "image.png"
        bad.write_bytes(b"\x89PNG\r\n")
        assert self.cli.main(["add", str(bad)]) == 1

    def test_ask_returns_answer_with_source(self, tmp_path, capsys):
        self.cli.main(["add", str(self._doc(tmp_path))])
        assert self.cli.main(["ask", "How many days of annual leave?"]) == 0
        out = capsys.readouterr().out
        assert "Sources" in out
        assert "handbook.txt" in out
        assert "25 days" in out

    def test_ask_unanswerable_exits_two(self, tmp_path):
        self.cli.main(["add", str(self._doc(tmp_path))])
        assert self.cli.main(["ask", "Who won the 1998 world cup final?"]) == 2

    def test_ask_with_empty_library_errors(self):
        assert self.cli.main(["ask", "anything"]) == 1

    def test_list_when_empty(self, capsys):
        assert self.cli.main(["list"]) == 0
        assert "empty" in capsys.readouterr().out.lower()


# --- answer construction ----------------------------------------------

class TestAnswering:
    def test_extractive_answer_quotes_and_cites(self, library):
        from docsassistant.answer import answer_question

        hits = library.search("meal allowance", top_k=3)
        result = answer_question("How much is the meal allowance?", hits, coverage=0.8)

        assert result.grounded is True
        assert result.citations
        assert result.citations[0].marker == 1
        assert "30 pounds" in result.text

    def test_refuses_below_coverage_floor(self, library):
        from docsassistant.answer import answer_question

        hits = library.search("meal allowance", top_k=3)
        result = answer_question("anything", hits, coverage=0.0)

        assert result.grounded is False
        assert result.citations == []

    def test_refuses_with_no_hits(self):
        from docsassistant.answer import answer_question

        result = answer_question("anything", [])
        assert result.grounded is False

    def test_blank_question_refused(self, library):
        from docsassistant.answer import answer_question

        hits = library.search("meal", top_k=1)
        assert answer_question("   ", hits, coverage=0.9).grounded is False

    def test_llm_failure_falls_back_to_quoting(self, library):
        """A provider outage must degrade to extractive, not 500."""
        from docsassistant.answer import answer_question

        class BrokenLLM:
            def complete(self, prompt: str) -> str:
                raise RuntimeError("provider unavailable")

        hits = library.search("meal allowance", top_k=3)
        result = answer_question(
            "How much is the meal allowance?", hits, llm=BrokenLLM(), coverage=0.8
        )
        assert result.grounded is True
        assert result.citations

    def test_llm_answer_is_used_when_it_works(self, library):
        from docsassistant.answer import answer_question

        class FakeLLM:
            def complete(self, prompt: str) -> str:
                return '{"answer": "The allowance is 30 pounds [1].", "confidence": "high"}'

        hits = library.search("meal allowance", top_k=3)
        result = answer_question(
            "How much?", hits, llm=FakeLLM(), coverage=0.8
        )
        assert "30 pounds" in result.text
        assert result.citations


# --- overview questions and greetings ---------------------------------

class TestOverviewAndGreetings:
    @pytest.mark.parametrize("question", [
        "What is this document about?",
        "summarise this",
        "give me a summary",
        "what are the key points",
        "what does this say",
        "tldr",
    ])
    def test_overview_questions_recognised(self, question):
        from docsassistant.answer import is_overview_question
        assert is_overview_question(question) is True

    @pytest.mark.parametrize("question", [
        "what is his email",
        "how many days of leave",
        "who is the manager",
    ])
    def test_specific_questions_not_treated_as_overview(self, question):
        from docsassistant.answer import is_overview_question
        assert is_overview_question(question) is False

    @pytest.mark.parametrize("question", ["hi", "Hello!", "thanks", "ok"])
    def test_greetings_recognised(self, question):
        from docsassistant.answer import is_greeting
        assert is_greeting(question) is True

    def test_question_containing_hi_is_not_a_greeting(self):
        from docsassistant.answer import is_greeting
        assert is_greeting("what is his hire date") is False

    def test_overview_question_is_answered_not_refused(self, client):
        """Regression: 'What is this document about?' has no distinctive
        terms, so coverage scored zero and it was refused — before the
        LLM was ever consulted."""
        client.post("/api/documents",
                    files=[("files", ("h.txt", DOC, "text/plain"))])
        answer = client.post(
            "/api/ask", json={"question": "What is this document about?"}
        ).json()
        assert answer["grounded"] is True
        assert answer["citations"]

    def test_greeting_gets_a_friendly_reply(self, client):
        client.post("/api/documents",
                    files=[("files", ("h.txt", DOC, "text/plain"))])
        answer = client.post("/api/ask", json={"question": "hi"}).json()
        assert answer["grounded"] is True
        assert answer["citations"] == []
        assert "ask me" in answer["answer"].lower()

    def test_leading_returns_opening_passages(self, library):
        hits = library.leading(limit=3)
        assert hits
        assert hits[0].chunk.position == 0
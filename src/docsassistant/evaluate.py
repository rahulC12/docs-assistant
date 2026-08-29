"""Measure retrieval quality instead of guessing at it.

Most RAG projects ship with no evaluation at all, so nobody — including
the author — knows whether a change made things better or worse. This
module provides a small labelled set and two standard metrics, which is
enough to catch a regression and enough to be honest in the README about
how well the system actually works.

Run it with:

    python -m docsassistant.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunk import chunk_document
from .extract import extract
from .index import Library


@dataclass
class EvalCase:
    """A question and the text that must appear in a retrieved chunk."""

    question: str
    must_contain: str
    note: str = ""


# The corpus below is synthetic on purpose: it can be published without
# leaking anything, and it exercises the cases that break naive systems
# — vocabulary mismatch, numbers, and facts split across sections.
CORPUS: dict[str, str] = {
    "handbook.txt": """Annual Leave

All full-time employees are entitled to 25 days of paid annual leave per
calendar year. Leave accrues monthly at a rate of 2.08 days per month of
service. Part-time employees accrue leave pro rata.

Requesting Leave

Leave requests must be submitted at least 14 days in advance through the
HR portal. Requests for more than 5 consecutive days require approval
from your line manager and the department head.

Carry Over

Employees may carry over a maximum of 5 unused days into the following
calendar year. Carried days must be used before 31 March or they are
forfeited.

Sick Leave

Sick leave is separate from annual leave. Employees receive 10 days of
paid sick leave per year. A doctor's note is required for absences
exceeding 3 consecutive days.

Remote Working

Employees may work remotely up to 3 days per week with manager approval.
Fully remote arrangements require a formal agreement reviewed annually.
""",
    "security.txt": """Password Requirements

Passwords must be at least 14 characters and include a mix of letters,
numbers and symbols. Passwords expire every 90 days and cannot be reused
within 12 changes.

Multi-Factor Authentication

MFA is mandatory for all accounts with access to production systems.
Hardware keys are issued on request and are required for administrator
accounts.

Incident Reporting

Suspected security incidents must be reported to the security team
within 1 hour of discovery. Do not attempt to investigate a suspected
breach yourself.

Data Classification

Documents are classified as Public, Internal, Confidential or Restricted.
Restricted material may not leave company-managed devices under any
circumstances.
""",
    "expenses.txt": """Travel Expenses

Economy class must be used for all flights under 6 hours. Business class
requires director approval and is permitted only for flights exceeding 6
hours.

Accommodation

Hotel costs are capped at 180 per night in major cities and 120
elsewhere. Receipts are required for all accommodation claims.

Meals

A daily meal allowance of 45 applies when travelling. Alcohol is not
reimbursable under any circumstances.

Submitting Claims

Expense claims must be submitted within 30 days of the expense being
incurred. Claims submitted after 60 days will not be reimbursed.
""",
}


CASES: list[EvalCase] = [
    # Direct vocabulary match — the easy baseline.
    EvalCase("How many days of annual leave do employees get?",
             "25 days"),
    EvalCase("What is the password minimum length?",
             "14 characters"),
    EvalCase("When must expense claims be submitted?",
             "within 30 days"),
    EvalCase("How much is the daily meal allowance?",
             "45"),

    # Vocabulary mismatch — the query uses different words to the source.
    EvalCase("How much holiday am I entitled to?",
             "25 days",
             "'holiday' never appears; the document says 'annual leave'"),
    EvalCase("Can I work from home?",
             "remotely",
             "'work from home' vs 'remote working'"),
    EvalCase("What happens if I am off sick for a week?",
             "doctor's note",
             "'off sick for a week' vs 'absences exceeding 3 days'"),

    # Facts that live in a specific section, easily confused with others.
    EvalCase("How many unused leave days can I carry over?",
             "maximum of 5 unused days"),
    EvalCase("Do I need approval for business class flights?",
             "director approval"),
    EvalCase("Who needs a hardware key?",
             "administrator accounts"),
    EvalCase("How quickly must a security incident be reported?",
             "within 1 hour"),
    EvalCase("What is the hotel cap outside major cities?",
             "120"),

    # Questions the corpus cannot answer — the system should retrieve
    # nothing convincing rather than confidently returning the closest
    # unrelated passage.
    EvalCase("What is the parental leave policy?", "",
             "not in the corpus — should return nothing relevant"),
    EvalCase("How do I request a company car?", "",
             "not in the corpus"),
]


def build_library() -> Library:
    library = Library()
    for filename, text in CORPUS.items():
        document = extract(filename, text.encode("utf-8"))
        chunks = chunk_document(document)
        library.add_document(
            document.doc_id, document.filename, chunks, document.page_count
        )
    return library


def evaluate(library: Library | None = None, top_k: int = 3) -> dict:
    """Return recall@k and mean reciprocal rank over the labelled set."""
    library = library or build_library()

    answerable = [c for c in CASES if c.must_contain]
    unanswerable = [c for c in CASES if not c.must_contain]

    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    failures: list[tuple[str, str]] = []

    for case in answerable:
        results = library.search(case.question, top_k=top_k)
        found_rank = None
        for rank, hit in enumerate(results, start=1):
            if case.must_contain.lower() in hit.chunk.text.lower():
                found_rank = rank
                break

        if found_rank == 1:
            hits_at_1 += 1
        if found_rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
            top = results[0].chunk.preview[:60] if results else "(nothing)"
            failures.append((case.question, top))

    # Refusal accuracy: does the answer layer decline exactly on the
    # questions the corpus cannot support? This matters more than recall
    # — a confident wrong answer is worse than no answer.
    from .answer import answer_question

    correct_refusals = 0
    false_refusals = 0

    for case in CASES:
        results = library.search(case.question, top_k=3)
        cov = library.coverage(case.question, results[0].chunk) if results else 0.0
        response = answer_question(case.question, results, coverage=cov)
        should_answer = bool(case.must_contain)
        if response.grounded == should_answer:
            if not should_answer:
                correct_refusals += 1
        elif should_answer:
            false_refusals += 1

    spurious = len(unanswerable) - correct_refusals

    total = len(answerable)
    return {
        "questions": total,
        "recall_at_1": round(hits_at_1 / total, 3),
        f"recall_at_{top_k}": round(hits_at_k / total, 3),
        "mrr": round(sum(reciprocal_ranks) / total, 3),
        "unanswerable": len(unanswerable),
        "correct_refusals": f"{correct_refusals}/{len(unanswerable)}",
        "false_refusals": false_refusals,
        "answered_when_it_should_not": spurious,
        "failures": failures,
    }


def main() -> None:  # pragma: no cover - manual tool
    library = build_library()
    print(f"corpus: {len(library.documents)} documents, "
          f"{library.chunk_count} chunks")
    print(f"retriever: {'hybrid' if library.is_hybrid else 'BM25 (lexical only)'}")
    print()

    results = evaluate(library)
    for key, value in results.items():
        if key == "failures":
            continue
        print(f"  {key:26} {value}")

    if results["failures"]:
        print("\n  questions where the answer was not in the top 3:")
        for question, top in results["failures"]:
            print(f"    - {question}")
            print(f"      best match: {top}…")


if __name__ == "__main__":  # pragma: no cover
    main()

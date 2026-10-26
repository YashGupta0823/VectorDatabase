"""Storage, persistence, de-duplication, and semantic retrieval tests."""

import re
import subprocess
import sys
from pathlib import Path

from conftest import FIXTURE_PAPERS
from database import SOURCE_ARXIV, SOURCE_NOTE, ResearchDatabase, paper_doc_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def test_store_and_read_back(loaded_db):
    assert loaded_db.count() == len(FIXTURE_PAPERS)
    doc = loaded_db.get("arxiv:test.0002")
    assert doc["metadata"]["title"] == FIXTURE_PAPERS[1]["title"]
    assert doc["metadata"]["authors"] == "Test Author B, Test Author C"
    assert doc["metadata"]["year"] == 2023
    assert doc["metadata"]["source_type"] == SOURCE_ARXIV
    assert doc["document"].startswith(FIXTURE_PAPERS[1]["title"])


def test_data_survives_new_client(db_path):
    first = ResearchDatabase(db_path)
    first.upsert_papers(FIXTURE_PAPERS[:2])
    first.add_note("Persisted idea", "Momentum in commodity futures after supply shocks.")
    del first

    reopened = ResearchDatabase(db_path)
    assert reopened.count() == 3
    assert reopened.count(SOURCE_NOTE) == 1


def test_data_survives_process_restart(db_path):
    """Write in this process, then read from a completely separate Python process."""
    db = ResearchDatabase(db_path)
    db.upsert_papers(FIXTURE_PAPERS)
    note_id = db.add_note("Restart check", "Volatility targeting for crypto portfolios.")

    script = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from database import ResearchDatabase;"
        "db = ResearchDatabase(sys.argv[2]);"
        "print(db.count(), db.get(sys.argv[3])['metadata']['title'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, str(PROJECT_ROOT), str(db_path), note_id],
        capture_output=True, text=True, timeout=180, check=True,
    ).stdout.strip().splitlines()[-1]
    assert out == f"{len(FIXTURE_PAPERS) + 1} Restart check"


def test_new_note_is_searchable_immediately(loaded_db):
    note_id = loaded_db.add_note(
        "Weather derivatives for agricultural hedging",
        "Using temperature-linked contracts to protect crop farmers against drought seasons.",
        url="https://example.org/note", author="Tester",
    )
    results = loaded_db.search("financial instruments that pay out when it doesn't rain", k=3)
    assert results[0].id == note_id
    assert results[0].is_note
    assert results[0].metadata["authors"] == "Tester"


def test_reimport_does_not_duplicate(db):
    added, skipped = db.upsert_papers(FIXTURE_PAPERS)
    assert (added, skipped) == (len(FIXTURE_PAPERS), 0)
    added, skipped = db.upsert_papers(FIXTURE_PAPERS)
    assert (added, skipped) == (0, len(FIXTURE_PAPERS))
    # a newer version of the same arXiv paper maps to the same ID
    v2 = dict(FIXTURE_PAPERS[0], arxiv_id="test.0001v2")
    assert db.upsert_papers([v2]) == (0, 1)
    assert db.count() == len(FIXTURE_PAPERS)


def test_same_note_twice_is_not_duplicated(db):
    a = db.add_note("Same", "Identical content")
    b = db.add_note("Same", "Identical content")
    assert a == b and db.count() == 1


def test_paper_doc_id_strips_versions():
    assert paper_doc_id("2610.32959v3") == "arxiv:2610.32959"
    assert paper_doc_id("2610.32959") == "arxiv:2610.32959"
    assert paper_doc_id("q-fin/0601001v1") == "arxiv:q-fin/0601001"


def test_semantic_retrieval_without_shared_keywords(loaded_db):
    cases = [
        ("Can a surge in demand tell us which direction an asset is heading?", "arxiv:test.0002"),
        ("Who pays when a storm wrecks a city, and can outside money absorb that?", "arxiv:test.0005"),
    ]
    for query, expected in cases:
        top = loaded_db.search(query, k=5)[0]
        assert top.id == expected, f"{query!r} -> {top.title}"
    # "collapse" protection is close to both put hedging and catastrophe bonds (also a
    # form of protection), so we only require the tail-risk paper in the top two and
    # ahead of the unrelated portfolio / pricing / microstructure fixtures.
    top2 = [r.id for r in loaded_db.search(
        "What protects my investments when everything collapses overnight?", k=5)[:2]]
    assert "arxiv:test.0001" in top2
    # sanity check that case 2 really shares no content words with its target
    target = loaded_db.get("arxiv:test.0002")["document"]
    content = _words(cases[0][0]) - {"can", "a", "in", "tell", "us", "which", "an", "is"}
    assert not content & _words(target)


def test_similarity_scores_are_ordered(loaded_db):
    results = loaded_db.search("option pricing models", k=5)
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in sims)


def test_source_type_filter(loaded_db):
    loaded_db.add_note("Note", "Order flow imbalance idea")
    results = loaded_db.search("order flow imbalance", k=10, source_type=SOURCE_ARXIV)
    assert results and all(not r.is_note for r in results)

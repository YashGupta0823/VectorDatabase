"""Research brief and idea-assessment tests."""

from research import (assess_idea, brief_to_markdown, data_period, detect_evidence,
                      detect_methods, export_brief, generate_brief, query_feedback,
                      split_sentences, word_used_in)

SECTIONS = [
    "## At a glance",
    "## 1. Research question",
    "## 2. Most relevant papers",
    "## 3. What each paper investigates",
    "## 4. How the papers connect to your question",
    "## 5. Similarities and differences",
    "## 6. Gaps and open questions",
    "## 7. Follow-up research questions",
    "## 8. References",
]


def test_brief_has_all_sections_in_order(loaded_db):
    question = "How can investors defend against extreme market losses?"
    brief = generate_brief(loaded_db, question, k=5)
    md = brief_to_markdown(brief)
    positions = [md.index(s) for s in SECTIONS]
    assert positions == sorted(positions)
    assert f"> {question}" in md
    assert 3 <= len(brief.digests) <= 5
    assert len(brief.follow_ups) == 3
    assert brief.confidence in {"High", "Moderate", "Low"}
    for d in brief.digests:
        assert d.result.url in md


def test_brief_quotes_are_verbatim_from_abstracts(loaded_db):
    brief = generate_brief(loaded_db, "predicting short-term price moves from trading activity")
    for d in brief.digests:
        assert d.aim in d.result.abstract
        assert d.best_passage in d.result.abstract
        for sentence in d.findings + d.caveats:
            assert sentence in d.result.abstract
        for tags in (d.methods, d.evidence, d.markets, d.practical):
            for phrase in tags.values():
                assert phrase.lower() in d.result.document.lower()


def test_brief_uses_only_arxiv_papers(loaded_db):
    loaded_db.add_note("My hedging note", "Buy puts to hedge crashes and limit drawdowns.")
    brief = generate_brief(loaded_db, "hedging crashes with puts")
    assert all(not d.result.is_note for d in brief.digests)


def test_brief_on_empty_database_returns_none(db):
    assert generate_brief(db, "anything") is None


def test_export_brief_writes_markdown(loaded_db, tmp_path):
    md = brief_to_markdown(generate_brief(loaded_db, "volatility surface calibration"))
    path = export_brief(md, "Volatility surface calibration?!", out_dir=tmp_path / "outputs")
    assert path.exists() and path.suffix == ".md"
    assert path.read_text(encoding="utf-8") == md
    assert "volatility-surface-calibration" in path.name


def test_assess_idea(loaded_db):
    a = assess_idea(loaded_db, "I want to test whether imbalances in the limit order book "
                               "predict next-second mid-price changes", k=3)
    assert a.digests[0].result.id == "arxiv:test.0002"
    assert a.band in {"Closely studied", "Related work exists", "Loosely related",
                      "Little related work in this corpus"}
    assert "Order book / market microstructure" in a.methods_to_explore


def test_assess_idea_empty_database(db):
    assert assess_idea(db, "some idea") is None


def test_split_sentences_keeps_abbreviations():
    s = split_sentences("We use e.g. GARCH models for forecasting. Results show gains of 5%. A third sentence is here.")
    assert s[0] == "We use e.g. GARCH models for forecasting."
    assert len(s) == 3


def test_detect_methods_reports_matched_phrase():
    found = detect_methods("A deep reinforcement learning agent for limit order book market making")
    assert found["Reinforcement learning"] == "reinforcement learning"
    assert "Order book / market microstructure" in found


def test_unused_query_words_really_are_unused(loaded_db):
    brief = generate_brief(loaded_db, "Can a surge in demand tell us which direction an asset is heading?")
    text = " ".join(d.result.document for d in brief.digests)
    for word in brief.unused_query_words:
        assert not word_used_in(word, text)
    # "hedging" in the question must not be reported as unused when an abstract says "hedging"
    brief = generate_brief(loaded_db, "hedge funds hedging crashes")
    assert "hedging" not in brief.unused_query_words


def test_trading_checks_only_for_trading_questions(loaded_db):
    trading = generate_brief(loaded_db, "Should I trade a strategy that buys puts before crashes?")
    other = generate_brief(loaded_db, "How do insurers share hurricane losses with capital markets?")
    assert trading.trading_question and not other.trading_question
    assert not any("transaction costs" in g for g in other.gaps)
    assert not any("transaction costs" in q for q in other.follow_ups)


def test_markdown_tables_escape_pipes(loaded_db):
    loaded_db.upsert_papers([{
        "arxiv_id": "test.pipe", "title": "A | B testing of hedging puts", "abstract":
        "We study hedging with put options and find lower drawdowns in equity crashes.",
        "authors": ["Test"], "published": "2024-01-01", "primary_category": "q-fin.RM",
        "categories": ["q-fin.RM"], "url": "https://example.org/pipe"}])
    md = brief_to_markdown(generate_brief(loaded_db, "hedging put options crashes"))
    assert "A \\| B testing" in md


def test_query_feedback_flags_broad_queries(loaded_db):
    results = loaded_db.search("momentum", k=3)
    assert any("short or broad" in t for t in query_feedback("momentum", results))
    long_q = "Does order-flow imbalance in the limit order book predict short-horizon returns?"
    assert not any("short or broad" in t for t in query_feedback(long_q, loaded_db.search(long_q, 3)))


def test_evidence_and_period_detection():
    text = "Using CRSP data from 2020 to 2025, we prove a theorem and run Monte Carlo simulations."
    assert set(detect_evidence(text)) == {"Empirical (real data)", "Theoretical / mathematical",
                                         "Simulation / numerical"}
    assert data_period(text) == "2020–2025"
    assert data_period("no dates here") == ""

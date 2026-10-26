"""CLI behaviour: empty/invalid input, empty database, long input, interrupts."""

import pytest

import app
from conftest import FIXTURE_PAPERS
from database import MAX_QUERY_CHARS, ResearchDatabase


def run(db_path, *args):
    return app.main(["--db", str(db_path), *args])


def feed_prompts(monkeypatch, answers):
    """Replace interactive prompts with scripted answers."""
    it = iter(answers)

    def next_answer(*a, **k):
        value = next(it)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(app.Prompt, "ask", next_answer)
    monkeypatch.setattr(app.Confirm, "ask", lambda *a, **k: False)


def test_empty_query_rejected(db_path):
    assert run(db_path, "search", "   ") == 2


def test_search_on_empty_database(db_path, capsys):
    assert run(db_path, "search", "volatility") == 0
    assert "database is empty" in capsys.readouterr().out


def test_fewer_results_than_requested(db_path, capsys):
    ResearchDatabase(db_path).upsert_papers(FIXTURE_PAPERS[:2])
    assert run(db_path, "search", "price impact", "-k", "5") == 0
    assert "Only 2 item(s) stored" in capsys.readouterr().out


def test_extremely_long_query(db_path):
    db = ResearchDatabase(db_path)
    db.upsert_papers(FIXTURE_PAPERS)
    results = db.search("hedging crash risk " * 5000, k=3)
    assert len(results) == 3
    assert MAX_QUERY_CHARS < len("hedging crash risk " * 5000)


def test_add_command_and_empty_note(db_path):
    assert run(db_path, "add", "--title", "Idea", "--content", "Carry trades in FX") == 0
    assert ResearchDatabase(db_path).count() == 1
    assert run(db_path, "add", "--title", "  ", "--content", "x") == 2


def test_count_and_stats(db_path, capsys):
    ResearchDatabase(db_path).upsert_papers(FIXTURE_PAPERS)
    assert run(db_path, "count") == 0
    assert run(db_path, "stats") == 0
    out = capsys.readouterr().out
    assert str(len(FIXTURE_PAPERS)) in out and "all-MiniLM-L6-v2" in out


def test_menu_invalid_choice_then_exit(db_path, monkeypatch, capsys):
    feed_prompts(monkeypatch, ["9", "abc", "6"])
    assert run(db_path) == 0
    out = capsys.readouterr().out
    assert "not a valid option" in out and "Goodbye" in out


def test_menu_keyboard_interrupt_exits_cleanly(db_path, monkeypatch, capsys):
    feed_prompts(monkeypatch, [KeyboardInterrupt()])
    assert run(db_path) == 0
    assert "Goodbye" in capsys.readouterr().out


def test_menu_ctrl_c_inside_action_returns_to_menu(db_path, monkeypatch, capsys):
    ResearchDatabase(db_path).upsert_papers(FIXTURE_PAPERS)
    feed_prompts(monkeypatch, ["1", KeyboardInterrupt(), "6"])
    assert run(db_path) == 0
    assert "Cancelled" in capsys.readouterr().out


def test_menu_empty_search_input_reprompts(db_path, monkeypatch, capsys):
    ResearchDatabase(db_path).upsert_papers(FIXTURE_PAPERS)
    feed_prompts(monkeypatch, ["1", "", "order flow imbalance", "", "6"])  # "" = pause
    assert run(db_path) == 0
    out = capsys.readouterr().out
    assert "Please enter some text" in out
    assert "Order-flow imbalance and short-horizon price impact" in out


@pytest.mark.parametrize("command", ["brief", "related"])
def test_brief_and_related_commands(db_path, command, capsys):
    ResearchDatabase(db_path).upsert_papers(FIXTURE_PAPERS)
    assert run(db_path, command, "protecting a portfolio from crashes", "-k", "3") == 0
    assert "Tail-risk hedging" in capsys.readouterr().out


def test_markup_like_text_is_displayed_literally(db_path, capsys):
    db = ResearchDatabase(db_path)
    db.add_note("Bounds on [0,1] and [/bold] tags", "Content with [red]brackets[/red] inside.")
    assert run(db_path, "search", "[/x] brackets", "-k", "1") == 0
    out = capsys.readouterr().out
    assert "[0,1]" in out and "[/bold]" in out and "[red]brackets[/red]" in out


def test_add_note_confirms_it_is_searchable(db_path, capsys):
    ResearchDatabase(db_path).upsert_papers(FIXTURE_PAPERS)
    assert run(db_path, "add", "--title", "Weather hedge", "--content",
               "Temperature-linked contracts that pay farmers during droughts") == 0
    out = capsys.readouterr().out
    assert "Searchable now" in out and "#1" in out


def test_menu_unexpected_error_is_friendly(db_path, monkeypatch, capsys):
    ResearchDatabase(db_path).upsert_papers(FIXTURE_PAPERS)
    monkeypatch.delenv("ALPHASEARCH_DEBUG", raising=False)

    def boom(db):
        raise RuntimeError("disk full")

    monkeypatch.setattr(app, "action_stats", boom)
    feed_prompts(monkeypatch, ["5", "6"])
    assert run(db_path) == 0
    out = capsys.readouterr().out
    assert "Something went wrong" in out and "disk full" in out and "Traceback" not in out


def test_menu_stats_then_pause_then_exit(db_path, monkeypatch, capsys):
    ResearchDatabase(db_path).upsert_papers(FIXTURE_PAPERS)
    feed_prompts(monkeypatch, ["5", "", "6"])
    assert run(db_path) == 0
    out = capsys.readouterr().out
    assert "A L P H A S E A R C H" in out and "Imported arXiv papers" in out and "Goodbye" in out

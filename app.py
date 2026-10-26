"""AlphaSearch — Semantic Research Intelligence for Quantitative Finance.

Run `python app.py` for the interactive menu, or `python app.py --help` for
non-interactive subcommands (import, count, stats, search, brief, related, add).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

from rich.console import Console, Group  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.prompt import Confirm, Prompt  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from database import (DEFAULT_DB_PATH, MAX_NOTE_CHARS, MAX_QUERY_CHARS, SOURCE_ARXIV,  # noqa: E402
                      SOURCE_NOTE, ResearchDatabase, SearchResult)
from load_data import (CATEGORIES, DEFAULT_TARGET, ArxivFetchError, import_papers,  # noqa: E402
                       read_ids_file)
from research import (SCORE_HELP, IdeaAssessment, assess_idea, brief_to_markdown,  # noqa: E402
                      category_name, export_brief, generate_brief, match_label, query_feedback,
                      short_authors)

console = Console()

MENU_ITEMS = [
    ("1", "Search Research", "Ask a question in plain English; get the closest papers"),
    ("2", "Generate Research Brief", "Structured 8-part brief from the top 3–5 papers"),
    ("3", "Add Research / Trading Idea", "Save your own note; it is searchable immediately"),
    ("4", "Find Related Research", "Check an idea against the existing literature"),
    ("5", "Database Statistics", "What is stored, where, and which model"),
    ("6", "Exit", ""),
]

EMPTY_DB_HELP = ("[yellow]The database is empty.[/] Load papers first with:\n"
                 "  [bold]python app.py import --ids-file corpus_ids.txt[/]   (the documented 600-paper "
                 "corpus; needs internet once)")

MATCH_STYLES = {"Strong": "green", "Good": "cyan", "Partial": "yellow", "Weak": "red"}


# --------------------------------------------------------------------- helpers
def excerpt(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def ensure_not_empty(db: ResearchDatabase) -> bool:
    if db.count() == 0:
        console.print(EMPTY_DB_HELP)
        return False
    return True


def ask_text(label: str, required: bool = True, max_chars: int | None = MAX_QUERY_CHARS) -> str:
    """Prompt for one line of text; re-asks on empty input when required."""
    while True:
        value = Prompt.ask(label, default="", show_default=False).strip()
        if not value and required:
            console.print("[yellow]Please enter some text (or press Ctrl+C to go back).[/]")
            continue
        if max_chars and len(value) > max_chars:
            console.print(f"[yellow]Input is {len(value)} characters; only the first {max_chars} "
                          f"will be used (the embedding model reads ~256 tokens anyway).[/]")
            value = value[:max_chars]
        return value


def ask_multiline(label: str) -> str:
    console.print(f"{label} [dim](paste or type; press Enter on an empty line to finish)[/]")
    lines = []
    while True:
        line = input()
        if not line.strip():
            if lines:
                break
            console.print("[yellow]Content cannot be empty.[/]")
            continue
        lines.append(line)
    return "\n".join(lines)


def pause() -> None:
    """Keep results on screen until the user is ready to see the menu again."""
    Prompt.ask("\n[dim]Press Enter to return to the main menu[/]", default="", show_default=False)


def score_text(sim: float) -> Text:
    """'████████░░  0.485  Strong match' with a colour per band."""
    label = match_label(sim)
    filled = max(0, min(10, round(sim * 10)))
    t = Text("█" * filled + "░" * (10 - filled), style=MATCH_STYLES[label])
    t.append(f"  {sim:.3f}  ", style="bold")
    t.append(f"{label} match", style=MATCH_STYLES[label])
    return t


def meta_line(r: SearchResult) -> str:
    if r.is_note:
        author = "" if r.authors == "User note" else f" · {escape(r.authors)}"
        return f"[magenta]User note[/]{author} · added {r.metadata.get('published', '')}"
    return (f"{escape(short_authors(r.authors))} · {r.metadata.get('published') or r.year} · "
            f"{escape(category_name(r.metadata.get('category', '')))} · arXiv:{r.metadata.get('arxiv_id', '')}")


def result_card(rank: int, r: SearchResult) -> Panel:
    body = Group(
        Text.from_markup(f"[bold]{escape(r.title)}[/]"),
        Text.from_markup(f"[dim]{meta_line(r)}[/]"),
        Text(""),
        Text(excerpt(r.abstract)),
        Text(""),
        score_text(r.similarity),
        Text.from_markup(f"[blue underline]{escape(r.url)}[/]") if r.url else Text("(no link)", style="dim"),
    )
    return Panel(body, title=f"[bold]#{rank}[/]", title_align="left",
                 border_style="magenta" if r.is_note else "cyan", padding=(0, 1))


def show_tips(tips: list[str]) -> None:
    for tip in tips:
        console.print(f"[yellow]Tip:[/] {escape(tip)}")


def show_results(results: list[SearchResult], query: str, requested: int) -> None:
    if not results:
        console.print("[yellow]No results found.[/]")
        return
    console.print(Rule(f"[bold]Top {len(results)} results[/]"))
    console.print(f"[bold]Query:[/] [italic]{escape(query)}[/]\n")
    if len(results) < requested:
        console.print(f"[dim](Only {len(results)} item(s) stored, fewer than the {requested} requested.)[/]")
    for rank, r in enumerate(results, 1):
        console.print(result_card(rank, r))
    console.print(f"[dim]{escape(SCORE_HELP)}[/]")
    show_tips(query_feedback(query, results))


def show_assessment(a: IdeaAssessment) -> None:
    color = {"Closely studied": "green", "Related work exists": "cyan",
             "Loosely related": "yellow"}.get(a.band, "red")
    top = a.digests[0].result.similarity
    summary = [f"[bold {color}]{a.band}[/] (best match {top:.3f})", escape(a.band_advice)]
    if a.search_terms:
        summary.append("\n[bold]The literature calls this:[/] "
                       + escape(", ".join(f'"{t}"' for t in a.search_terms)))
    if a.markets:
        summary.append("[bold]Markets studied:[/] " + escape("; ".join(
            f"{m} ({', '.join(f'#{i + 1}' for i in idx)})" for m, idx in a.markets.items())))
    if a.evidence:
        summary.append("[bold]Evidence:[/] " + escape("; ".join(
            f"{e} ({', '.join(f'#{i + 1}' for i in idx)})" for e, idx in a.evidence.items())))
    summary.append("\n[dim]Heuristic: based on similarity to the stored corpus and phrases in the "
                   "abstracts, not on a review of the full literature.[/]")
    console.print(Panel("\n".join(summary), title="[bold]Your idea vs. existing literature[/]",
                        title_align="left", border_style=color, padding=(0, 1)))

    for i, d in enumerate(a.digests):
        r = d.result
        shared = ", ".join(a.shared_terms.get(i, [])) or "none (matched on meaning only)"
        body = Group(
            Text.from_markup(f"[bold]{escape(r.title)}[/]"),
            Text.from_markup(f"[dim]{meta_line(r)}[/]"),
            score_text(r.similarity),
            Text(""),
            Text.from_markup(f"[bold]Closest passage to your idea:[/] \"{escape(d.best_passage)}\""),
            Text.from_markup(f"[bold]Words shared with your idea:[/] {escape(shared)}"),
            Text.from_markup(f"[bold]Methods / topics:[/] {escape(', '.join(d.methods) or '—')}"),
            Text.from_markup(f"[blue underline]{escape(r.url)}[/]"),
        )
        console.print(Panel(body, title=f"[bold]#{i + 1}[/]", title_align="left",
                            border_style="cyan", padding=(0, 1)))

    if a.methods_to_explore:
        table = Table(title="Approaches used in related work")
        table.add_column("Method / topic")
        table.add_column("Papers")
        for m, idx in a.methods_to_explore.items():
            table.add_row(m, ", ".join(f"#{i + 1}" for i in idx))
        console.print(table)
    if a.uncovered_idea_terms:
        console.print("[bold]Words in your idea that none of these papers use[/] "
                      "[dim](possible new angle, or just different wording)[/]: "
                      + escape(", ".join(a.uncovered_idea_terms)))
    show_tips(a.tips)


# --------------------------------------------------------------------- actions
def action_search(db: ResearchDatabase, query: str | None = None, k: int = 5) -> str | None:
    if not ensure_not_empty(db):
        return None
    if query is None:
        console.print("[dim]Describe what you want to know in plain English, e.g. "
                      "\"How do insurers pass disaster risk to investors?\"[/]")
        query = ask_text("[bold]Research question[/]")
    if len(query) > MAX_QUERY_CHARS:
        console.print(f"[yellow]Query is {len(query):,} characters; only the first {MAX_QUERY_CHARS:,} "
                      f"are used (the embedding model reads ~256 tokens anyway).[/]")
    with console.status("Embedding query and searching…"):
        results = db.search(query, k=k)
    show_results(results, query, k)
    return query


def action_brief(db: ResearchDatabase, question: str | None = None, k: int = 5,
                 export: bool | None = None) -> None:
    if not ensure_not_empty(db):
        return
    if question is None:
        console.print("[dim]Describe a strategy or research question, e.g. \"I want to sell options "
                      "to harvest the volatility risk premium. What does the research say?\"[/]")
        question = ask_text("[bold]Research question for the brief[/]")
    with console.status("Retrieving papers and building brief…"):
        brief = generate_brief(db, question, k=k)
    if brief is None:
        console.print("[yellow]No arXiv papers in the database to build a brief from. "
                      "Run `python app.py import` first.[/]")
        return
    markdown = brief_to_markdown(brief)
    console.print(Panel(Markdown(markdown), border_style="cyan", padding=(1, 2)))
    if export is None:
        export = Confirm.ask("Export this brief to a Markdown file in outputs/?", default=False)
    if export:
        path = export_brief(markdown, question)
        console.print(f"[green]✓ Saved brief to[/] {escape(str(path))}")


def action_add(db: ResearchDatabase, title: str | None = None, content: str | None = None,
               url: str | None = None, author: str | None = None) -> str:
    interactive = content is None
    if interactive and title is None:
        console.print("[dim]Add a paper summary, trading idea or research note. It is embedded and "
                      "saved to the database immediately.[/]")
    title = title or ask_text("[bold]Title[/]", max_chars=500)
    content = content if content is not None else ask_multiline("[bold]Content / abstract / idea[/]")
    if interactive:
        url = ask_text("Source URL [dim](optional, press Enter to skip)[/]", required=False)
        author = ask_text("Author [dim](optional, press Enter to skip)[/]", required=False)
    if len(content) > MAX_NOTE_CHARS:
        console.print(f"[yellow]Content is long; only the first {MAX_NOTE_CHARS:,} characters are stored, "
                      f"and the embedding reflects roughly the first 256 tokens.[/]")
    with console.status("Embedding and saving…"):
        doc_id = db.add_note(title, content, url=url or "", author=author or "")
        # prove it is retrievable right away by searching for its own content
        hits = db.search(content, k=5)
    rank = next((i for i, r in enumerate(hits, 1) if r.id == doc_id), None)
    check = (f"[green]✓ Searchable now:[/] a search for its content returns it at #{rank}."
             if rank else "[yellow]Saved, but it is not in the top 5 for its own content.[/]")
    console.print(Panel(
        f"[green]✓ Saved[/] [bold]{escape(title)}[/]\n"
        f"ID: {doc_id} · Notes stored: {db.count(SOURCE_NOTE)} · Total items: {db.count()}\n"
        f"{check}\nIt is stored on disk in {escape(str(db.path))} and will still be there after a restart.",
        title="[bold]Research note added[/]", title_align="left", border_style="green", padding=(0, 1)))
    return doc_id


def action_related(db: ResearchDatabase, idea: str | None = None, k: int = 5) -> None:
    if not ensure_not_empty(db):
        return
    interactive = idea is None
    if interactive:
        console.print("[dim]Describe your idea, e.g. \"I want to investigate whether liquidity shocks "
                      "can predict intraday volatility.\"[/]")
        idea = ask_text("[bold]Your trading or research idea[/]")
    with console.status("Comparing your idea with stored papers…"):
        assessment = assess_idea(db, idea, k=k)
    if assessment is None:
        console.print("[yellow]No arXiv papers stored yet. Run `python app.py import` first.[/]")
        return
    show_assessment(assessment)
    if interactive and Confirm.ask("Save this idea as a research note?", default=False):
        title = ask_text("[bold]Short title for the idea[/]", max_chars=500)
        action_add(db, title=title, content=idea, url="", author="")


def action_stats(db: ResearchDatabase) -> None:
    s = db.stats()
    table = Table(title="AlphaSearch database", show_header=False, title_style="bold cyan")
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Total stored items", str(s["total"]))
    table.add_row("Imported arXiv papers", str(s["arxiv_papers"]))
    table.add_row("User-added research notes", str(s["user_notes"]))
    table.add_row("Database location", s["path"])
    table.add_row("Collection", s["collection"])
    table.add_row("Embedding model", s["embedding_model"])
    table.add_row("Distance metric", s["distance"])
    console.print(table)


def action_import(db: ResearchDatabase, limit: int, categories: list[str] | None,
                  from_cache: bool, ids_file: str | None = None) -> int:
    try:
        ids = read_ids_file(ids_file) if ids_file else None
        summary = import_papers(db, target=limit, categories=categories, from_cache=from_cache,
                                ids=ids, progress=lambda msg: console.print(msg, highlight=False))
    except ArxivFetchError as exc:
        console.print(f"[red]Import failed:[/] {escape(str(exc))}\n"
                      "Check your internet connection and run the import again. It is safe to "
                      "rerun and will skip papers already stored.\n"
                      f"Papers currently stored: {db.count()}")
        return 1
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        return 1
    console.print(f"[green]Import complete.[/] Fetched {summary['fetched']}, "
                  f"added {summary['added']} new, skipped {summary['skipped_existing']} already "
                  f"stored. Total items in database: [bold]{summary['total_in_db']}[/]")
    return 0


# ------------------------------------------------------------------------ menu
def show_banner(db: ResearchDatabase) -> None:
    papers, notes = db.count(SOURCE_ARXIV), db.count(SOURCE_NOTE)
    title = Text("A L P H A S E A R C H", style="bold cyan", justify="center")
    subtitle = Text("Semantic Research Intelligence for Quantitative Finance", style="italic",
                    justify="center")
    status = Text(f"{papers} arXiv papers · {notes} research note{'s' if notes != 1 else ''} · "
                  f"local & offline-ready",
                  style="dim", justify="center")
    console.print(Panel(Group(title, subtitle, Text(""), status), border_style="cyan", padding=(1, 2)))
    if papers + notes == 0:
        console.print(EMPTY_DB_HELP)


def show_menu() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2), title="[bold]Main menu[/]",
                  title_justify="left")
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="bold")
    table.add_column(style="dim")
    for key, name, desc in MENU_ITEMS:
        table.add_row(f"[{key}]", name, desc)
    console.print()
    console.print(table)


def interactive_menu(db: ResearchDatabase) -> None:
    show_banner(db)
    actions = {
        "1": lambda: _search_then_offer_brief(db),
        "2": lambda: action_brief(db),
        "3": lambda: action_add(db),
        "4": lambda: action_related(db),
        "5": lambda: action_stats(db),
    }
    while True:
        show_menu()
        try:
            choice = Prompt.ask("\nChoose an option [dim](1–6)[/]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if choice in ("6", "q", "quit", "exit"):
            break
        action = actions.get(choice)
        if not choice:
            console.print("[yellow]Type a number from 1 to 6 and press Enter.[/]")
            continue
        if action is None:
            console.print(f"[yellow]'{escape(choice[:20])}' is not a valid option. Enter a number from 1 to 6.[/]")
            continue
        try:
            action()
            pause()
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled. Back to the main menu.[/]")
        except EOFError:
            break
        except ValueError as exc:
            console.print(f"[yellow]{escape(str(exc))}[/]")
        except Exception as exc:  # keep the session alive; never show a stack trace mid-demo
            if os.environ.get("ALPHASEARCH_DEBUG"):
                raise
            console.print(f"[red]Something went wrong:[/] {escape(str(exc))}\n"
                          "[dim]Your data is safe. Set ALPHASEARCH_DEBUG=1 to see full details.[/]")
    console.print(f"\n[cyan]Goodbye. Your data is saved in[/] {escape(str(db.path))}")


def _search_then_offer_brief(db: ResearchDatabase) -> None:
    query = action_search(db)
    if query and db.count(SOURCE_ARXIV) and Confirm.ask(
            "\nGenerate a research brief for this question?", default=False):
        action_brief(db, question=query)


# ------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alphasearch", description=__doc__.split("\n")[0])
    p.add_argument("--db", default=str(DEFAULT_DB_PATH), help="ChromaDB directory (default: ./chroma_db)")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("menu", help="interactive menu (default)")
    imp = sub.add_parser("import", help="fetch arXiv q-fin papers into the database")
    imp.add_argument("--limit", type=int, default=DEFAULT_TARGET, help=f"papers to fetch (default {DEFAULT_TARGET})")
    imp.add_argument("--categories", nargs="+", choices=list(CATEGORIES), help="subset of categories")
    imp.add_argument("--from-cache", action="store_true", help="rebuild from data/arxiv_papers.json (offline)")
    imp.add_argument("--ids-file", help="fetch the exact arXiv IDs listed in this file (e.g. corpus_ids.txt)")
    sub.add_parser("count", help="print the number of stored documents")
    sub.add_parser("stats", help="database statistics")

    for name, text in (("search", "semantic search"), ("brief", "generate a research brief"),
                       ("related", "find research related to your idea")):
        sp = sub.add_parser(name, help=text)
        sp.add_argument("text", nargs="+", help="query / question / idea")
        sp.add_argument("-k", type=int, default=5, help="number of papers (default 5)")
        if name == "brief":
            sp.add_argument("--export", action="store_true", help="save Markdown to outputs/")

    add = sub.add_parser("add", help="add a research note")
    add.add_argument("--title", required=True)
    add.add_argument("--content", required=True)
    add.add_argument("--url", default="")
    add.add_argument("--author", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with console.status("Opening database…"):
            db = ResearchDatabase(args.db)
        cmd = args.command or "menu"
        text = " ".join(getattr(args, "text", None) or []).strip()
        if cmd in ("search", "brief", "related") and not text:
            raise ValueError("Please provide non-empty text.")
        if cmd == "menu":
            interactive_menu(db)
        elif cmd == "import":
            return action_import(db, max(1, args.limit), args.categories, args.from_cache, args.ids_file)
        elif cmd == "count":
            console.print(db.count())
        elif cmd == "stats":
            action_stats(db)
        elif cmd == "search":
            action_search(db, text, k=max(1, min(args.k, 50)))
        elif cmd == "brief":
            action_brief(db, text, k=args.k, export=args.export)
        elif cmd == "related":
            action_related(db, text, k=max(1, min(args.k, 20)))
        elif cmd == "add":
            action_add(db, args.title, args.content, args.url, args.author)
        return 0
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        return 130
    except ValueError as exc:
        console.print(f"[red]Error:[/] {escape(str(exc))}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

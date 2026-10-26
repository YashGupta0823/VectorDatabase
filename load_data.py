"""Fetch real quantitative-finance papers from the arXiv API and store them.

arXiv API docs: https://info.arxiv.org/help/api/user-manual.html
Terms of use ask clients to wait ~3 seconds between requests, which we honour.

Fetched papers are cached to data/arxiv_papers.json so the vector database
can be rebuilt offline with `python app.py import --from-cache`.
"""

from __future__ import annotations

import json
import math
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import requests

from database import PROJECT_ROOT, ResearchDatabase

ARXIV_API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "AlphaSearch/1.0 (student project; semantic research search)"
REQUEST_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 4
PAGE_SIZE = 100

DEFAULT_TARGET = 600
CATEGORIES = {
    "q-fin.TR": "Trading and Market Microstructure",
    "q-fin.PM": "Portfolio Management",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.PR": "Pricing of Securities",
    "q-fin.CP": "Computational Finance",
}

DATA_DIR = PROJECT_ROOT / "data"
CACHE_FILE = DATA_DIR / "arxiv_papers.json"

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

Progress = Callable[[str], None]


class ArxivFetchError(RuntimeError):
    """Raised when the arXiv API cannot be reached after all retries."""


# --------------------------------------------------------------------- parsing
def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_feed(xml_text: str) -> tuple[list[dict], int]:
    """Parse an arXiv Atom feed into paper dicts. Returns (papers, total_results)."""
    root = ET.fromstring(xml_text)
    total = int(root.findtext("opensearch:totalResults", default="0", namespaces=_NS) or 0)
    papers = []
    for entry in root.findall("atom:entry", _NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=_NS)
        title = _clean(entry.findtext("atom:title", namespaces=_NS))
        abstract = _clean(entry.findtext("atom:summary", namespaces=_NS))
        if not raw_id or not title or not abstract:
            continue  # arXiv occasionally returns error entries; never invent content
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        primary = entry.find("arxiv:primary_category", _NS)
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": [_clean(a.findtext("atom:name", namespaces=_NS))
                        for a in entry.findall("atom:author", _NS)],
            "published": (entry.findtext("atom:published", default="", namespaces=_NS) or "")[:10],
            "primary_category": primary.get("term", "") if primary is not None else "",
            "categories": [c.get("term", "") for c in entry.findall("atom:category", _NS)],
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return papers, total


# -------------------------------------------------------------------- fetching
def _describe(exc: Exception | None) -> str:
    """Readable error text; some network errors stringify to an empty string."""
    if exc is None:
        return "unknown error"
    text = " ".join(str(exc).split())
    if len(text) > 120:
        text = text[:117] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _request(session: requests.Session, params: dict, label: str,
             progress: Progress = print) -> tuple[list[dict], int]:
    """One arXiv API call with timeouts and exponential-backoff retries."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(ARXIV_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code} from arXiv")
            resp.raise_for_status()
            papers, total = parse_feed(resp.text)
            # arXiv sometimes returns a transiently empty page; retry it.
            if not papers and params.get("start", 0) < total:
                raise requests.HTTPError("arXiv returned an empty page; retrying")
            return papers, total
        except (requests.RequestException, ET.ParseError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                wait = REQUEST_DELAY_SECONDS * 2 ** (attempt - 1)
                progress(f"  ! {label} attempt {attempt} failed: {_describe(exc)}. "
                         f"Retrying in {wait:.0f}s…")
                time.sleep(wait)
    raise ArxivFetchError(f"Could not fetch {label} from arXiv after {MAX_RETRIES} attempts: "
                          f"{_describe(last_error)}")


def fetch_page(session: requests.Session, category: str, start: int, size: int = PAGE_SIZE,
               progress: Progress = print) -> tuple[list[dict], int]:
    """Fetch one page of a category, newest submissions first."""
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return _request(session, params, f"{category} (start={start})", progress)


def fetch_by_ids(ids: list[str], progress: Progress = print, session: requests.Session | None = None,
                 delay: float = REQUEST_DELAY_SECONDS) -> list[dict]:
    """Fetch an exact list of arXiv IDs (used to reproduce a saved corpus)."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    papers: list[dict] = []
    for i in range(0, len(ids), PAGE_SIZE):
        if i:
            time.sleep(delay)
        chunk = ids[i:i + PAGE_SIZE]
        page, _ = _request(session, {"id_list": ",".join(chunk), "max_results": len(chunk)},
                           f"IDs {i + 1}-{i + len(chunk)}", progress)
        papers.extend(page)
        progress(f"  ✓ fetched {len(papers)}/{len(ids)} papers by ID")
    return papers


def fetch_papers(target: int = DEFAULT_TARGET, categories: list[str] | None = None,
                 progress: Progress = print, session: requests.Session | None = None,
                 delay: float = REQUEST_DELAY_SECONDS, max_pages_per_category: int = 5) -> list[dict]:
    """Collect ~target unique papers, split evenly across the categories.

    Cross-listed papers appear in several categories; we keep paging a
    category until it contributes its share of *new* unique papers.
    """
    categories = categories or list(CATEGORIES)
    per_category = math.ceil(target / len(categories))
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)

    collected: dict[str, dict] = {}
    first_request = True
    for category in categories:
        added_here, start = 0, 0
        for _ in range(max_pages_per_category):
            if added_here >= per_category:
                break
            if not first_request:
                time.sleep(delay)  # respectful pacing between every request
            first_request = False
            page, total = fetch_page(session, category, start, progress=progress)
            for paper in page:
                if added_here >= per_category:
                    break
                if paper["arxiv_id"] not in collected:
                    collected[paper["arxiv_id"]] = paper
                    added_here += 1
            start += len(page)
            if not page or start >= total:
                break
        progress(f"  ✓ {category:9} +{added_here:4} new papers   (total unique: {len(collected)})")
        if len(collected) >= target:
            break
    return list(collected.values())[:target]


# --------------------------------------------------------------------- caching
def read_ids_file(path: Path) -> list[str]:
    """One arXiv ID per line; blank lines and # comments are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.split("#", 1)[0].strip() for ln in lines if ln.split("#", 1)[0].strip()]


def write_ids_file(papers: list[dict], path: Path) -> None:
    header = "# arXiv IDs of the AlphaSearch corpus. Rebuild with: python app.py import --ids-file corpus_ids.txt\n"
    Path(path).write_text(header + "\n".join(p["arxiv_id"] for p in papers) + "\n", encoding="utf-8")



def save_cache(papers: list[dict], path: Path = CACHE_FILE) -> None:
    """Merge papers into the local JSON cache (keyed by arXiv ID)."""
    existing = {p["arxiv_id"]: p for p in load_cache(path)}
    for p in papers:
        existing[p["arxiv_id"]] = p
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(existing.values()), indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_cache(path: Path = CACHE_FILE) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- import
def import_papers(db: ResearchDatabase, target: int = DEFAULT_TARGET,
                  categories: list[str] | None = None, from_cache: bool = False,
                  progress: Progress = print, cache_path: Path = CACHE_FILE,
                  ids: list[str] | None = None) -> dict:
    """Fetch (or read cached) papers and store them in the vector database.

    Raises ArxivFetchError on network failure; nothing is written in that case
    for the failed request, and rerunning the import is always safe.
    """
    if from_cache:
        papers = load_cache(cache_path)
        if not papers:
            raise FileNotFoundError(f"No cached papers at {cache_path}. Run an online import first.")
        papers = papers[:target]
        progress(f"Loaded {len(papers)} papers from cache {cache_path}")
    elif ids:
        progress(f"Fetching {len(ids)} specific papers from arXiv by ID…")
        papers = fetch_by_ids(ids, progress=progress)
        save_cache(papers, cache_path)
        progress(f"Fetched {len(papers)} papers; cached to {cache_path}")
    else:
        progress(f"Fetching ~{target} papers from arXiv ({REQUEST_DELAY_SECONDS:.0f}s between requests)…")
        papers = fetch_papers(target, categories, progress=progress)
        save_cache(papers, cache_path)
        progress(f"Fetched {len(papers)} unique papers; cached to {cache_path}")

    progress("Embedding and storing papers (first run downloads the ~80 MB embedding model)…")
    added, skipped = db.upsert_papers(
        papers, on_batch=lambda done, total: progress(f"  embedded {done}/{total} new papers"))
    return {"fetched": len(papers), "added": added, "skipped_existing": skipped, "total_in_db": db.count()}

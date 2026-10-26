"""ChromaDB persistence layer for AlphaSearch.

Every stored item (an arXiv paper or a user-added note) is one Chroma
document. The searchable text is "title + abstract/content"; everything else
lives in metadata. Embeddings come from Chroma's default local model
(all-MiniLM-L6-v2 via ONNX), so no API keys or network access are needed
once the model has been downloaded on first use.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path(os.environ.get("ALPHASEARCH_DB", PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = "alphasearch_research"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2 (Chroma default, ONNX, 384 dimensions)"

SOURCE_ARXIV = "arxiv"
SOURCE_NOTE = "user_note"

# all-MiniLM-L6-v2 only reads the first ~256 tokens, so very long queries add
# nothing but cost. Notes may be longer; only their beginning is embedded.
MAX_QUERY_CHARS = 2000
MAX_NOTE_CHARS = 20000
MAX_TITLE_CHARS = 500


@dataclass
class SearchResult:
    """One retrieved document with its cosine distance to the query."""

    id: str
    document: str
    metadata: dict
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity (1 - cosine distance); higher is more relevant."""
        return 1.0 - self.distance

    @property
    def title(self) -> str:
        return self.metadata.get("title", "(untitled)")

    @property
    def abstract(self) -> str:
        """Document text without the leading title line."""
        title = self.metadata.get("title", "")
        text = self.document
        if title and text.startswith(title):
            text = text[len(title):]
        return text.strip()

    @property
    def authors(self) -> str:
        return self.metadata.get("authors") or "Unknown"

    @property
    def year(self) -> str:
        return str(self.metadata.get("year") or "n.d.")

    @property
    def url(self) -> str:
        return self.metadata.get("url") or ""

    @property
    def is_note(self) -> bool:
        return self.metadata.get("source_type") == SOURCE_NOTE


class ResearchDatabase:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_function = DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            configuration={"hnsw": {"space": "cosine"}},
        )

    # ------------------------------------------------------------------ writes
    def upsert_papers(self, papers: list[dict], batch_size: int = 50,
                      on_batch: Callable[[int, int], None] | None = None) -> tuple[int, int]:
        """Store arXiv papers, skipping any whose ID already exists.

        Returns (added, skipped). IDs are derived from the version-less arXiv
        ID, so re-importing the same paper can never create a duplicate.
        """
        unique: dict[str, dict] = {}
        for paper in papers:
            unique.setdefault(paper_doc_id(paper["arxiv_id"]), paper)

        ids = list(unique)
        existing = set()
        for i in range(0, len(ids), 500):
            existing.update(self.collection.get(ids=ids[i:i + 500], include=[])["ids"])

        new = [(doc_id, p) for doc_id, p in unique.items() if doc_id not in existing]
        for i in range(0, len(new), batch_size):
            chunk = new[i:i + batch_size]
            self.collection.upsert(
                ids=[doc_id for doc_id, _ in chunk],
                documents=[f"{p['title']}\n\n{p['abstract']}" for _, p in chunk],
                metadatas=[paper_metadata(p) for _, p in chunk],
            )
            if on_batch:
                on_batch(min(i + batch_size, len(new)), len(new))
        skipped = len(papers) - len(new)
        return len(new), skipped

    def add_note(self, title: str, content: str, url: str = "", author: str = "") -> str:
        """Add a user-written research note or trading idea; returns its ID.

        The ID is a hash of title + content, so adding the exact same note
        twice updates it instead of duplicating it.
        """
        title = " ".join(title.split())[:MAX_TITLE_CHARS]
        content = content.strip()[:MAX_NOTE_CHARS]
        if not title or not content:
            raise ValueError("Both a title and content are required.")
        digest = hashlib.sha1(f"{title}\n{content}".encode("utf-8")).hexdigest()[:16]
        doc_id = f"note:{digest}"
        today = date.today()
        self.collection.upsert(
            ids=[doc_id],
            documents=[f"{title}\n\n{content}"],
            metadatas=[{
                "source_type": SOURCE_NOTE,
                "title": title,
                "authors": author.strip() or "User note",
                "published": today.isoformat(),
                "year": today.year,
                "category": "user-note",
                "url": url.strip(),
                "arxiv_id": "",
            }],
        )
        return doc_id

    # ------------------------------------------------------------------- reads
    def search(self, query: str, k: int = 5, source_type: str | None = None) -> list[SearchResult]:
        """Embed the query and return the k nearest documents by cosine distance."""
        query = " ".join(query.split())[:MAX_QUERY_CHARS]
        if not query:
            raise ValueError("Query is empty.")
        total = self.count()
        if total == 0 or k <= 0:
            return []
        where = {"source_type": source_type} if source_type else None
        res = self.collection.query(
            query_texts=[query],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(id=i, document=d, metadata=m or {}, distance=float(dist))
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get(self, doc_id: str) -> dict | None:
        res = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        return {"id": res["ids"][0], "document": res["documents"][0], "metadata": res["metadatas"][0]}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed arbitrary texts with the same model used for the collection."""
        return [list(map(float, v)) for v in self.embedding_function(texts)]

    def count(self, source_type: str | None = None) -> int:
        if source_type is None:
            return self.collection.count()
        return len(self.collection.get(where={"source_type": source_type}, include=[])["ids"])

    def stats(self) -> dict:
        return {
            "total": self.count(),
            "arxiv_papers": self.count(SOURCE_ARXIV),
            "user_notes": self.count(SOURCE_NOTE),
            "path": str(self.path.resolve()),
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "distance": "cosine",
        }


def paper_doc_id(arxiv_id: str) -> str:
    """Stable Chroma ID for an arXiv paper, ignoring the version suffix."""
    base = arxiv_id.strip()
    if "v" in base.rsplit("/", 1)[-1]:
        head, _, tail = base.rpartition("v")
        if tail.isdigit():
            base = head
    return f"arxiv:{base}"


def paper_metadata(paper: dict) -> dict:
    published = paper.get("published", "")
    return {
        "source_type": SOURCE_ARXIV,
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "authors": ", ".join(paper.get("authors", [])) or "Unknown",
        "published": published,
        "year": int(published[:4]) if published[:4].isdigit() else 0,
        "category": paper.get("primary_category", ""),
        "categories": ", ".join(paper.get("categories", [])),
        "url": paper.get("url", ""),
    }

"""Reproduce the README's semantic-search examples against the local database.

For each example query this prints:
  * the top-5 semantic results (rank, similarity, title, link);
  * a keyword-search baseline over the same corpus, ranked by how many of the
    query's meaningful words a document contains. The baseline is deliberately
    generous: case-insensitive, suffix-stripped and prefix-matched, so
    "investors" also matches "investment" and "bill" matches "billion";
  * for each target paper discussed in the README: its semantic rank and its
    keyword rank (or "not found").

Nothing is hard-coded except the queries and the IDs of the papers the README
discusses; every rank and score is computed live. Run from the repo root:

    python scripts/verify_semantic_examples.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import SOURCE_ARXIV, ResearchDatabase  # noqa: E402

EXAMPLES = [
    {
        "name": "Example 1: catastrophe risk transfer",
        "query": "Who covers the bill when a hurricane strikes, and can outside investors carry that danger?",
        "targets": ["arxiv:2609.22052", "arxiv:2610.03793"],  # the two catastrophe-bond papers
    },
    {
        "name": "Example 2: bank runs",
        "query": "What happens when frightened customers all rush to withdraw their savings at once?",
        "targets": ["arxiv:2608.11266"],  # Bank Run Exposure ... Loss-Averse Depositors
    },
    {
        "name": "Alternate: order flow and price direction",
        "query": "Can a surge in demand tell us which direction an asset is heading?",
        "targets": ["arxiv:2607.01377", "arxiv:2609.07989"],  # signed order flow papers
    },
]

STOPWORDS = set("""a an the and or of to in on at by for with from is are was were be been can could who what
when which how do does did that this those these there their them they us we my me i you your it its as
into if will would should may might than then so whether about after all any just more most much not no
only other over some such very once one get gets got make makes made keep kept""".split())
SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ies", "ied", "ed", "es", "s", "ly",
            "ments", "ment")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def stem(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def query_terms(query: str) -> list[str]:
    return sorted({w for w in tokens(query) if w not in STOPWORDS and len(w) > 2})


def term_matches(term: str, doc_tokens: set[str]) -> bool:
    s = stem(term)
    return any(stem(d) == s or (len(s) >= 4 and d.startswith(s)) for d in doc_tokens)


def keyword_ranking(query: str, corpus: dict[str, str]) -> dict[str, tuple[int, list[str]]]:
    """doc id -> (rank, matched terms) for documents matching at least one term."""
    terms = query_terms(query)
    scored = []
    for doc_id, text in corpus.items():
        doc_tokens = set(tokens(text))
        matched = [t for t in terms if term_matches(t, doc_tokens)]
        if matched:
            scored.append((len(matched), doc_id, matched))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return {doc_id: (rank, matched) for rank, (_, doc_id, matched) in enumerate(scored, 1)}


def main() -> None:
    db = ResearchDatabase()
    data = db.collection.get(where={"source_type": SOURCE_ARXIV}, include=["documents", "metadatas"])
    corpus = dict(zip(data["ids"], data["documents"]))
    titles = {i: m["title"] for i, m in zip(data["ids"], data["metadatas"])}
    print(f"Corpus: {len(corpus)} arXiv papers\n")

    for ex in EXAMPLES:
        query = ex["query"]
        kw = keyword_ranking(query, corpus)
        semantic = db.search(query, k=len(corpus), source_type=SOURCE_ARXIV)
        sem_rank = {r.id: (n, r.similarity) for n, r in enumerate(semantic, 1)}

        print("=" * 100)
        print(f"{ex['name']}\nQUERY: {query}")
        print(f"Meaningful query terms: {', '.join(query_terms(query))}")
        print("\nTop-5 semantic results:")
        for n, r in enumerate(semantic[:5], 1):
            k = kw.get(r.id)
            k_text = f"keyword rank {k[0]} (matched: {', '.join(k[1])})" if k else "keyword search: not found"
            print(f"  #{n}  {r.similarity:.3f}  {r.title}\n        {r.url}  |  {k_text}")
        print(f"\nKeyword baseline: {len(kw)} of {len(corpus)} papers match at least one term; "
              f"{sum(1 for v in kw.values() if len(v[1]) == len(query_terms(query)))} match all terms.")
        print("Target papers discussed in the README:")
        for t in ex["targets"]:
            if t not in corpus:
                print(f"  {t}: NOT IN DATABASE (import with --ids-file corpus_ids.txt)")
                continue
            s_rank, sim = sem_rank[t]
            k = kw.get(t)
            if k:
                n = len(k[1])
                above = sum(1 for v in kw.values() if len(v[1]) > n)
                tied = sum(1 for v in kw.values() if len(v[1]) == n) - 1
                k_text = (f"keyword search matches only {n} of {len(query_terms(query))} terms "
                          f"({', '.join(k[1])}); {above} papers match more terms and {tied} tie with it")
            else:
                k_text = "keyword search: NOT FOUND (shares no meaningful query term)"
            print(f"  {titles[t]}\n    semantic rank {s_rank} (similarity {sim:.3f}) | {k_text}")
        print()


if __name__ == "__main__":
    main()

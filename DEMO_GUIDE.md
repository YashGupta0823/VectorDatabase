# AlphaSearch: 2–3 Minute Demo Guide

A script for a short screen recording. Every query and input below was run against the
600-paper corpus (`corpus_ids.txt`) on 2026-10-26. The expected results shown are what that run
produced. If your database was built from a different corpus, rankings will differ.

Each step lists the suggested time, what to type, and what to point out.

---

## Before you record (not on camera)

```bash
cd VectorDatabase
source .venv/bin/activate
python app.py stats          # expect: 600 imported arXiv papers
python -m pytest -q          # optional sanity check: all tests pass
clear
```

- Make the terminal wide (at least ~100 columns) and increase the font size so the cards are
  readable.
- If `stats` shows 0 papers, run `python app.py import --ids-file corpus_ids.txt` first (needs
  internet; about 40 seconds on the development machine).

---

## 1. Launch (~10 s)

```bash
python app.py
```

**Point out:** the banner shows 600 arXiv papers stored locally, and the app works offline.

## 2. Semantic search (~35 s)

Type `1`, then paste:

```
Who covers the bill when a hurricane strikes, and can outside investors carry that danger?
```

**Expected:** #1 is *Design and pricing of a transparent parametric-modeled loss CAT bond:
application to German windstorm* (similarity 0.485, "Good match"). #3 is *Optimal Catastrophe
Bond Design and its Applications to Climate Change Risk* (0.426).

**Point out:**
- The query never says "catastrophe bond", "reinsurance" or "capital markets", yet the
  catastrophe-bond papers come first.
- The word "hurricane" appears in **none** of the 600 papers, and the #1 paper contains none
  of the query's meaningful words. Keyword search could not find it.
- The similarity bar and legend: about 0.1–0.25 is typical for unrelated papers, so 0.485
  stands out.

Answer `n` to the brief prompt, then press Enter to return to the menu.

## 3. Research brief (~45 s)

Type `2`, then paste:

```
I want to sell options to harvest the volatility risk premium. What does the research say?
```

**Expected:** Retrieval confidence **High**. #1 is *Harvesting the Variance Risk Premium in
Nuclear and Energy Equities: A Short-Put Portfolio Derisking Strategy* (0.530).

Scroll and **point out:**
- **At a glance:** recurring themes, evidence base, markets studied, and "Search with the
  literature's vocabulary".
- **Section 3:** quotes are verbatim from the abstracts, with tags showing the phrase that
  triggered them.
- **Section 6 (Gaps):** the authors' own caveat for [1], "…performance is measured before
  transaction costs and with a fixed universe". That's exactly what a trader needs to know.
- **Section 7:** three follow-up research questions.
- No language model is used. Everything is extracted or computed, and heuristics are labelled.

Answer `y` to export, show the saved path (`outputs/brief_….md`), then press Enter.

## 4. Add a new research note (~25 s)

Type `3` and enter:

| Prompt | Type |
|---|---|
| Title | `Weekend gap reversal in bitcoin` |
| Content | `Hypothesis: large Saturday-Sunday price gaps in bitcoin partially reverse by Monday close, because liquidity is thin on weekends and market makers widen spreads.` then press Enter on an empty line |
| Source URL | press Enter (skip) |
| Author | press Enter, or type your name |

**Expected:** a green "Research note added" panel with "✓ Searchable now: a search for its
content returns it at #1."

Press Enter to return to the menu.

## 5. Find the new note with different wording (~20 s)

Type `1`, then paste:

```
Do cryptocurrency moves over the weekend get undone once regular trading resumes?
```

**Expected:** #1 is your note *Weekend gap reversal in bitcoin* (magenta "User note" card).
Real papers on crypto market efficiency and short-horizon mean reversion follow in #2–#3.

**Point out:**
- The note is searchable immediately, with no re-indexing step.
- The query doesn't use "gap", "reversal", "Saturday" or "Monday". (It does share
  "weekend", so this shows meaning-based matching, not zero overlap.)

Answer `n`, press Enter, then type `6` to exit.

## 6. Persistence after restart (~20 s)

```bash
python app.py
```

**Point out:** the banner now says "600 arXiv papers · **1 research note**". Type `5` to show
the statistics table (601 items, 1 user note, the database path, and the embedding model).
Optionally repeat the search from step 5 to show the note is still #1. Then exit with `6`.

---

## What the audience should take away about semantic search

- **Meaning, not words.** Each paper and each query becomes a 384-number vector
  (all-MiniLM-L6-v2). Search returns the papers whose vectors point in the most similar
  direction (cosine similarity).
- **Plain-English questions find academic jargon.** "Hurricane… outside investors" finds
  "catastrophe bonds… capital markets". Keyword search fails because the important words don't
  overlap.
- **Scores are honest.** Weak matches are labelled Weak or Partial, and very short queries get
  a tip rather than confident-looking results.

## Why ChromaDB fits this project

- **Local and persistent.** `PersistentClient` stores vectors, text and metadata on disk in
  `chroma_db/`, so data survives restarts and works offline. No server, account or API key.
- **Embeddings built in.** Its default embedding function runs all-MiniLM-L6-v2 locally (ONNX on
  CPU), so adding a note or searching is one call.
- **Metadata filtering.** Each item stores its source type, authors, date, category and URL. The
  brief uses a `source_type = arxiv` filter, so only real papers are cited.
- **Right-sized.** A few hundred to a few thousand short documents is well within a single-machine
  HNSW index. Search returns in well under a second on a laptop, and upserts with stable IDs
  make re-imports duplicate-free.

---

## After recording (optional cleanup)

The demo note is real data and will stay in your database. To remove it:

```bash
python -c "
from database import ResearchDatabase
db = ResearchDatabase()
ids = db.collection.get(where={'title': 'Weekend gap reversal in bitcoin'})['ids']
db.collection.delete(ids=ids); print('deleted', ids, '- total items now', db.count())"
```

Delete any exported demo briefs from `outputs/` if you don't want them.

## If something goes wrong on camera

- Typo in a menu choice: the app says so and shows the menu again.
- Wrong query: press **Ctrl+C** to cancel and return to the main menu.
- Results scrolled away: each action waits for **Enter** before showing the menu again.

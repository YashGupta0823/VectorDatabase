# AlphaSearch — Semantic Research Intelligence for Quantitative Finance

A command-line research assistant that finds quantitative-finance papers by **meaning** rather than
by keyword, using a local ChromaDB vector database of real arXiv abstracts, and then turns the
results into a structured, citation-grounded research brief.

> **Note to self (delete this block before submitting):** Sections 1–4 must be written in my own
> words. Each one has a ✍️ placeholder followed by factual notes to work from. The notes are
> reference material, not text to paste. Sections 5 onward are technical documentation.

---

## 1. What AlphaSearch does and why I chose this concept

> ✍️ **TODO — write in your own words.**

*Facts to draw on:*
- Five features: semantic search, a research brief generator, adding your own research notes,
  finding related research for an idea, and database statistics.
- A user types a question in plain English, and the app returns arXiv papers whose title and
  abstract are closest in embedding space.
- The research brief does not use an LLM. It quotes abstract sentences verbatim, matches method
  phrases, and compares the papers using term overlap and embedding similarity. Interpretive
  parts are labelled *heuristic*.
- Your motivation: why quant finance? Why is keyword search a problem here? (Academic jargon
  such as "order-flow imbalance" or "catastrophe bond" doesn't match how practitioners or
  students phrase questions.)

## 2. What the database stores and how it is organized

> ✍️ **TODO — write in your own words.**

*Facts to draw on:*
- One persistent ChromaDB collection, `alphasearch_research`, stored in `./chroma_db/`.
- **One document = one paper (or one user note).** Document text = `title + "\n\n" + abstract`.
  Abstracts are short (typically one paragraph), so the model's ~256-token window covers most of
  each one. Chunking wasn't needed, and one vector per paper keeps citations simple.
- Metadata on every item: `source_type` (`arxiv` or `user_note`), `arxiv_id`, `title`,
  `authors`, `published` (YYYY-MM-DD), `year`, `category` (arXiv *primary* category),
  `categories` (all cross-listings), and `url`.
- IDs: `arxiv:<id-without-version>` for papers (e.g. `arxiv:2609.22052`), so re-imports and
  newer versions of a paper never duplicate it. User notes get `note:<sha1(title+content)[:16]>`,
  so adding the same note twice doesn't duplicate it either.
- Corpus as imported on 2026-10-26: **600 unique papers**, 100 newest from each of q-fin.TR, PM, RM,
  ST, PR, and CP. Because arXiv cross-lists papers, some have a primary category outside q-fin
  (e.g. cs.AI, econ.EM, stat.ML), but every one is listed in at least one q-fin category.
- Similarity is cosine. The app shows `similarity = 1 − cosine distance`.

## 3. How users add information and search

> ✍️ **TODO — write in your own words.**

*Facts to draw on:*
- Menu **[1] Search Research**: enter a question and see the top 5 papers. Each card shows
  the title, authors, publication date, category, arXiv ID, a ~300-character abstract excerpt,
  a similarity bar with a Strong/Good/Partial/Weak label, and the link. A one-line legend
  explains the score. Very short queries or weak matches trigger a tip. You're then offered a
  brief for the same question.
- Menu **[2] Generate Research Brief**: an "At a glance" summary plus 8 sections built from
  the top 3–5 arXiv papers:
  1. question
  2. ranked paper table
  3. what each paper investigates
  4. how the papers connect to the question
  5. similarities and differences
  6. gaps
  7. three follow-up questions
  8. references

  It can export Markdown to `outputs/`. See section 9.
- Menu **[3] Add Research / Trading Idea**: title, content (multi-line), optional URL and
  author. The note is embedded and saved immediately. The app then searches for the note's
  own content and reports its rank (e.g. "returns it at #1"), so the user sees it is
  searchable. It persists after restart.
- Menu **[4] Find Related Research**: describe an idea. You get a coverage rating (e.g.
  "Closely studied"), the phrases the literature uses for your idea, the markets and evidence
  types involved, the closest passage in each paper, and words from your idea that none of the
  papers use. You can then save the idea as a note.
- Menu **[5] Database Statistics**.
- After each action, "Press Enter to return to the main menu". Ctrl+C cancels the current
  action.
- Every feature also exists as a non-interactive subcommand (see section 8).

## 4. Semantic search examples: why keyword search fails

> ✍️ **TODO — explain each example in your own words**, using the verified results below.
> Reproduce them at any time with `python scripts/verify_semantic_examples.py`.

**Verified on 2026-10-26 against the 600-paper corpus in `corpus_ids.txt`** (model
all-MiniLM-L6-v2, cosine similarity). Re-verified after rebuilding the corpus from scratch on a
clean setup: identical rankings and scores.

**How the keyword baseline works** (in `scripts/verify_semantic_examples.py`). It takes the
query's meaningful words (stopwords removed) and searches all 600 titles and abstracts. It's
deliberately *generous* to keyword search:
- matching is case-insensitive;
- suffixes are stripped ("investors" → "invest");
- prefixes match ("invest" matches "investment"; "bill" matches "billion");
- papers are ranked by how many query words they contain.

"Not found" means a paper contains **none** of the query's meaningful words, even under
those rules.

### Example 1: catastrophe risk transfer

**Query:** `Who covers the bill when a hurricane strikes, and can outside investors carry that danger?`
Meaningful terms: bill, carry, covers, danger, hurricane, investors, outside, strikes.

| Rank | Similarity | Paper | Keyword search |
|---|---|---|---|
| **1** | **0.485** | [Design and pricing of a transparent parametric-modeled loss CAT bond: application to German windstorm](https://arxiv.org/abs/2609.22052) | **not found** |
| 2 | 0.444 | [Agentic AI Systems and Financial Stability, From Model Risk to Systemic Risk](https://arxiv.org/abs/2610.08806) | not found |
| **3** | **0.426** | [Optimal Catastrophe Bond Design and its Applications to Climate Change Risk](https://arxiv.org/abs/2610.03793) | matches 1 of 8 terms ("bill", via "Billion-Dollar"); 16 papers match more terms, 112 tie |
| 4 | 0.426 | [The Missile and the Toll: Two-Way Flow Through the Strait of Hormuz](https://arxiv.org/abs/2610.18823) | not found |
| 5 | 0.416 | [Renewing Reliability: Valuation and Credit Risk Adjustments for Renewable Power Purchase Agreements](https://arxiv.org/abs/2607.04781) | matches "covers" |

*Facts to draw on:*
- The corpus's two catastrophe-bond papers rank #1 and #3. Their abstracts say (verbatim):
  "Catastrophe (cat) bonds overcome some lack of reinsurance by sourcing capacity from the
  wider capital markets" (#1) and "Catastrophe bonds have become a central instrument for
  transferring catastrophic risk to capital markets" (#3). That is the academic phrasing of
  "outside investors carry that danger".
- Keyword search can't find #1 at all. #3 matches only through "bill" (inside "NOAA's
  Billion-Dollar Weather and Climate Disasters database"), so it sits among 113 papers tied at
  one matching term. "hurricane" appears in 0 of the 600 papers; the papers say "windstorm",
  "natural disasters" and "catastrophic risk". No paper contains all 8 terms.
- Weak results, honestly: #2 (systemic risk from AI) and #5 (credit risk in power-purchase
  agreements) are only loosely related through general "who bears the risk" language. #4 isn't
  relevant.

### Example 2: bank runs

**Query:** `What happens when frightened customers all rush to withdraw their savings at once?`
Meaningful terms: customers, frightened, happens, rush, savings, withdraw.

| Rank | Similarity | Paper | Keyword search |
|---|---|---|---|
| **1** | **0.401** | [Bank Run Exposure in a Paycheck-to-Paycheck Economy with Loss-Averse Depositors](https://arxiv.org/abs/2608.11266) | **not found** |
| 2 | 0.366 | [Exponential investors with weakly mean-reverting prices](https://arxiv.org/abs/2610.08631) | not found |
| 3 | 0.349 | [Market Efficiency in a Changing Asset Population: Mobility, Entry, and Survival over Cryptocurrency's Full History](https://arxiv.org/abs/2610.23283) | not found |
| 4 | 0.348 | [RetailAgent: Structured Adverse Timing in Self-Conditioned Multimodal LLM Trading Agents](https://arxiv.org/abs/2608.28399) | not found |
| 5 | 0.344 | [Measuring the engine of a liquidation cascade: subcritical branching inside a first-order transition](https://arxiv.org/abs/2608.03616) | matches "withdraw" |

*Facts to draw on:*
- The query describes a **bank run** without using "bank", "run" or "deposit". The #1 paper's
  abstract begins (verbatim): "We develop a behavioural model of bank run exposure in a
  paycheck-to-paycheck economy with loss averse depositors", and says high subjective
  bad-state probabilities "endogenously increase liquidity demand and generate equilibrium
  stress states supporting bank runs."
- Keyword search can't find it: it contains none of the six query terms ("customers",
  "savings" and "withdraw" don't appear; it says "depositors", "deposits" and "bank runs").
  Only 25 of 600 papers match any query term at all.
- Weak results, honestly: only #1 is on topic. #2–#5 score 0.34–0.37 ("Partial") and are
  unrelated or loosely related (#5 is a liquidation cascade). The score gap between #1 and #2
  reflects that.
- Wording matters. "What happens when frightened customers all rush to pull their money out at
  the same time?" did **not** put this paper in the top 5. "What if everyone panics and tries
  to get their deposits back on the same day?" ranked it #1 (0.371), but that wording shares
  the word "deposits" with it. Worth mentioning as a limitation.

### Alternate example: order flow and price direction

**Query:** `Can a surge in demand tell us which direction an asset is heading?` Top 5:
Herding and Liquidity in Order-Book Markets III (0.417), Modelling Regime Shifts in Continuous
Intraday Electricity Markets with State-dependent Hawkes Processes (0.388), **Regimes in the
Order Flow** (0.375), **Liquidity Premium and Investment Horizons** (0.367), The Reconfiguration
Premium (0.364).
- None of the top 5 contains any of the six query terms, even with generous matching.
- The most on-point paper is #4 ("Signed order flow strongly predicts contemporaneous and
  one-month-ahead returns", verbatim), but it's only ranked 4th. That's why Example 2 above
  is the stronger demonstration.

### A suggested query that did *not* work well

`What protects my investments when everything collapses overnight?` was the planned first
example. On this corpus its top results were systemic-risk and liquidation-cascade papers
(similarities 0.34–0.37), not tail-risk hedging. The 600 newest q-fin papers contain few
portfolio-insurance papers, and the vague phrasing pulled in "systemic collapse" topics.

---

## 5. Data source and reproducible import

- **Source:** the [arXiv API](https://info.arxiv.org/help/api/user-manual.html)
  (`https://export.arxiv.org/api/query`), categories q-fin.TR, q-fin.PM, q-fin.RM, q-fin.ST,
  q-fin.PR, and q-fin.CP. All titles, abstracts, and authors come from arXiv. Nothing is
  generated.
- **Politeness:** 3-second delay between requests (per arXiv's API terms), 30 s timeout, up to
  4 attempts with exponential backoff on network errors, HTTP 429/5xx, or transiently empty pages.
- **Failure handling:** if arXiv can't be reached, the import stops with a short error (e.g.
  `ProxyError: ...`), exits with code 1, reports how many papers are still stored, and stores
  nothing from the failed run. Rerunning is always safe. Tested by routing traffic through a
  dead proxy.
- **Duplicates:** IDs come from the version-less arXiv ID, and existing IDs are skipped. Running
  the import repeatedly never creates duplicates (verified: re-import reported
  `added 0, skipped 30`).
- **Cache:** fetched papers are saved to `data/arxiv_papers.json` (git-ignored), so the database
  can be rebuilt offline.

```bash
# Default: newest ~600 papers, split evenly across the six categories (6 API calls)
python app.py import

# Exact reproduction of the corpus used in this README (600 IDs, committed in the repo)
python app.py import --ids-file corpus_ids.txt

# Smaller / custom imports
python app.py import --limit 60
python app.py import --limit 200 --categories q-fin.TR q-fin.RM

# Rebuild the vector DB offline from data/arxiv_papers.json
python app.py import --from-cache

# Check the document count
python app.py count
```

Because the default import takes the *newest* papers, running it on a later date gives a
different corpus. Use `--ids-file corpus_ids.txt` to reproduce the examples above.

**Reproducibility check (2026-10-26).** I copied only the files Git would track into an empty
directory, then set up a fresh venv and an empty home directory (so the embedding model had to
be downloaded again) and ran `pip install -r requirements.txt` and
`python app.py import --ids-file corpus_ids.txt`. Result: all 600 papers re-fetched from
arXiv. Every document and metadata field was identical to the original database, both README
examples returned the same top 5 with the same scores, and the test suite passed.

## 6. Vector database and embedding model

| Component | Choice |
|---|---|
| Vector DB | [ChromaDB](https://docs.trychroma.com/) `PersistentClient`, tested with chromadb 1.5.9 |
| Storage | `./chroma_db/` (override with `--db PATH` or the `ALPHASEARCH_DB` env var) |
| Collection | `alphasearch_research`, HNSW index, `space = cosine` |
| Embeddings | Chroma's `DefaultEmbeddingFunction`: **all-MiniLM-L6-v2** (ONNX, 384-dim), runs locally on CPU |
| Score shown | `similarity = 1 − cosine distance` (higher = closer in meaning) |
| Telemetry | Chroma anonymized telemetry disabled |

The same model also embeds individual abstract sentences when building a brief, to find the
passage closest to the user's question.

### Reading similarity scores

Scores from this model are lower than people expect, so the app labels them. Across several
sample queries on this corpus, the *median* paper scored 0.07–0.25 (i.e. unrelated), and
clearly on-topic matches scored roughly 0.40–0.63.

| Label | Similarity | Meaning |
|---|---|---|
| Strong | ≥ 0.50 | close topical match |
| Good | 0.40–0.50 | clearly related |
| Partial | 0.30–0.40 | adjacent topic; check the abstract |
| Weak | < 0.30 | probably unrelated, or the corpus lacks this topic |

Scores are only comparable within this model and corpus. A broad query like "finance" can
still score 0.48, because broadness isn't the same as low similarity. That's why the app also
flags queries with fewer than four words.

## 7. Setup and first-run downloads

Requirements: Python 3.11+ (developed and tested on **Python 3.13.3**, macOS). No API keys and no
paid services.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py import --ids-file corpus_ids.txt   # or: python app.py import
```

First-run downloads:
1. Python packages from `requirements.txt` (ChromaDB pulls in onnxruntime, tokenizers, etc.).
2. The **all-MiniLM-L6-v2 ONNX model (~79 MB)**, downloaded automatically by ChromaDB on first
   use to `~/.cache/chroma/onnx_models/`.
3. Paper metadata from arXiv (6 requests for the default import, 3 seconds apart).

After that, everything (search, briefs, notes) works **offline**. This was verified by running a
search with all HTTP/HTTPS traffic routed to a dead proxy.

Measured on the development Mac (2026-10-26):
- `pip install -r requirements.txt` into a fresh venv: ~23 s
- `import --ids-file corpus_ids.txt` including the first-time model download: ~39 s
- re-embedding all 600 papers from cache: ~8 s

## 8. Running the application

```bash
python app.py                 # interactive menu
```

```
╭──────────────────────────────────────────────────────────────╮
│                    A L P H A S E A R C H                     │
│   Semantic Research Intelligence for Quantitative Finance    │
│  600 arXiv papers · 0 research notes · local & offline-ready │
╰──────────────────────────────────────────────────────────────╯
Main menu
  [1]  Search Research              Ask a question in plain English; get the closest papers
  [2]  Generate Research Brief      Structured 8-part brief from the top 3–5 papers
  [3]  Add Research / Trading Idea  Save your own note; it is searchable immediately
  [4]  Find Related Research        Check an idea against the existing literature
  [5]  Database Statistics          What is stored, where, and which model
  [6]  Exit
```

For a scripted 2–3 minute walkthrough, see [`DEMO_GUIDE.md`](DEMO_GUIDE.md).

Non-interactive subcommands:

```bash
python app.py search "How can traders price uncertainty around future volatility?" -k 5
python app.py brief  "I want to sell options to harvest the volatility risk premium. What does the research say?" --export
python app.py related "I want to investigate whether liquidity shocks can predict intraday volatility"
python app.py add --title "VIX carry idea" --content "Short front-month VIX futures in contango..." \
                  --url "https://example.com" --author "Your Name"
python app.py stats
python app.py count
python app.py --help
```

Exported briefs are written to `outputs/brief_<timestamp>_<slug>.md` (git-ignored).

Edge cases handled:
- empty database: prints import instructions;
- empty input: re-prompts, or exits with code 2;
- invalid or blank menu choices;
- fewer than *k* stored items;
- very short or broad queries, and weak best matches: shows a tip;
- very long input: truncated to 2,000 characters with a warning, since the model reads only
  ~256 tokens;
- Ctrl+C: cancels the current action and returns to the menu; at the menu prompt it exits;
- failed or repeated imports;
- unexpected errors during the menu: a one-line message instead of a stack trace (set
  `ALPHASEARCH_DEBUG=1` to see the trace).

## 9. How the research brief works (no LLM)

A brief is generated from the top 3–5 **arXiv** papers for the question. User notes are excluded,
so every citation is a real paper. Each part is produced deterministically:

| Brief part | How it is produced |
|---|---|
| At a glance | Retrieval confidence (High/Moderate/Low from the score bands), recurring methods, evidence mix, markets, which paper to read first, and the literature's own vocabulary for the topic |
| 1. Research question | The user's text |
| 2. Most relevant papers | Table: title (linked), year, category, similarity, match label |
| 3. What each paper investigates | **Verbatim** abstract sentences: the aim (cue phrases like "we propose", "this paper"; if none is found the brief says so), reported results ("we find", "results show"), and any caveat the authors state ("before transaction costs", "remains open"). Tags for evidence type (empirical / theoretical / simulation), markets (equities, options, crypto, …), data period (year ranges), and methods, each shown with the abstract phrase that triggered it |
| 4. How they connect to the question | For each paper, the abstract sentence whose embedding is closest to the question. It also lists question words that appear in no retrieved abstract, i.e. matched by meaning only (checked with prefix/stem matching so the claim is never false) |
| 5. Similarities and differences | A comparison matrix (evidence × markets × methods), shared methods/markets/vocabulary, distinctive terms per paper, and the closest and most different pair by embedding similarity. Labelled **heuristic** |
| 6. Gaps and open questions | Stated caveats, plus things *no abstract mentions*: for trading questions, transaction costs, out-of-sample tests and risk reporting; missing evidence types; single-market evidence; low retrieval confidence. Worded as "not mentioned in the abstract", not "not done" |
| 7. Follow-up questions | Exactly three, chosen by priority from the gaps (costs, out-of-sample, market transfer, method transfer, shared-theme robustness, replication) |
| 8. References | Author (year), title, arXiv ID, link |

Limitations:
- It reads abstracts only, not full papers.
- Cue-phrase extraction can pick a less informative sentence.
- The phrase lists are hand-written and incomplete; a missing tag means "not detected", not
  "absent".
- Trading-specific checks run only when the question looks like a trading question (words like
  trade, strategy, returns, predict, hedge).

## 10. Testing

```bash
python -m pytest -q
```

46 tests, each using a temporary ChromaDB directory (the real `chroma_db/` is never touched).
They cover:
- storing and reading back items;
- persistence across a new client **and a separate Python process**;
- immediate searchability of new notes, including the in-app confirmation;
- no duplicates on re-import (including `v2` versions);
- semantic retrieval with no shared keywords;
- brief structure, exactly three follow-ups, and verbatim quotes, caveats and tag phrases;
- that "unused question words" are really unused;
- trading checks applied only to trading questions;
- broad-query feedback;
- Markdown table escaping and Markdown export;
- arXiv XML parsing, and retry/failure handling with a mocked network (no real requests);
- CLI edge cases: empty input, invalid menu choices, Ctrl+C, empty database, long input,
  markup-like text, and friendly handling of unexpected errors.

Test fixtures in `tests/conftest.py` are short **synthetic** texts written for testing only and
are clearly labelled as such.

## 11. Project structure

```
app.py                  CLI: interactive menu + argparse subcommands (Rich UI)
database.py             ChromaDB PersistentClient wrapper: upsert, notes, search, stats
load_data.py            arXiv API client: paging, retries, pacing, caching, import
research.py             Research brief + idea assessment (deterministic, grounded)
corpus_ids.txt          arXiv IDs of the 600-paper corpus (for exact reproduction)
scripts/verify_semantic_examples.py   Reproduces section 4 with a generous keyword baseline
DEMO_GUIDE.md           Script for a 2–3 minute screen recording
tests/                  pytest suite (temporary DBs, mocked network)
prompt_log.txt          AI usage log
data/  chroma_db/  outputs/   local, git-ignored
```

## 12. Troubleshooting

- **`The database is empty`**: run `python app.py import` (needs internet once).
- **Import fails / times out**: arXiv may be rate-limiting or down. Wait a minute and rerun.
  Already-stored papers are skipped.
- **Slow first search**: the embedding model is downloading (~79 MB) or loading.
- **Want a clean slate**: delete the `chroma_db/` folder and re-import (from cache with
  `--from-cache` if `data/` exists).
- **Python version**: ChromaDB 1.5.x supports 3.13. Developed on 3.13.3. Older Pythons
  (≥3.11) should work, but only 3.13 was tested.
- **One-word queries give odd results** (e.g. "momentum"): sentence-embedding models need
  context. Describe the mechanism and market in a full sentence; the app shows a tip when a
  query is very short.

### Known limitations

- Abstracts only: the brief never reads full papers.
- The corpus is the 600 newest q-fin papers as of 2026-10-26. Classic topics (e.g. portfolio
  insurance) are thinly covered.
- Similarity scores depend on wording. Paraphrases of the same question can reorder results
  (see Example 2).
- Single technical terms retrieve poorly compared with full-sentence questions.

## 13. AI usage disclosure

> ✍️ **Review and edit this to reflect your actual process.**

- **Claude Code (Anthropic, Claude Opus 5.5)** was the primary coding assistant. It generated
  the initial implementation, tests, and technical documentation from my specification, and ran
  the imports and tests.
- **ChatGPT (GPT-6)** was used for project ideation and planning before implementation.
- The full prompts, development steps, and my manual review are logged in
  [`prompt_log.txt`](prompt_log.txt).
- Sections 1–4 of this README were written by me.

## 14. Citations and links

- arXiv API User's Manual: https://info.arxiv.org/help/api/user-manual.html
- arXiv API Terms of Use: https://info.arxiv.org/help/api/tou.html. *Thank you to arXiv for use
  of its open access interoperability.*
- ChromaDB documentation: https://docs.trychroma.com/
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks.* EMNLP. https://arxiv.org/abs/1908.10084
- Wang, W. et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression
  of Pre-Trained Transformers.* NeurIPS. https://arxiv.org/abs/2002.10957
- Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Rich (terminal UI): https://github.com/Textualize/rich
- Papers cited in section 4 are linked in the tables above. All paper content belongs to its authors.

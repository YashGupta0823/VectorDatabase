"""Research brief generation and idea assessment — no LLM required.

Everything here is deterministic and grounded in the retrieved abstracts:

* "Aim" / "reported result" are verbatim abstract sentences picked by cue
  phrases ("we propose", "we find", ...). If no cue is found, the brief says so.
* "Closest passage" is the abstract sentence whose embedding is closest to the
  user's question (same MiniLM model as the vector search).
* Methods, evidence type, markets and practical checks (transaction costs,
  out-of-sample testing, ...) come from fixed phrase lexicons. The matched
  phrase is always kept so a reader can verify it against the abstract.
* Comparisons, gaps and follow-up questions are computed from those tags,
  term overlap and embedding similarity. They are labelled as heuristics and
  phrased as "mentioned / not mentioned in the abstracts", never as claims
  about what the full papers contain.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from database import PROJECT_ROOT, SOURCE_ARXIV, ResearchDatabase, SearchResult

OUTPUT_DIR = PROJECT_ROOT / "outputs"

STOPWORDS = set("""
a about above across after again against all almost along also although always am among an and
another any are around as at based be because been before being below between both but by can
could did do does doing done due during each either else enough especially etc even ever every
few for from further given had has have having here how however i if in into is it its itself
just least less like made make makes many may might more most much must my namely near nearly
neither no nor not now of off often on once one only onto or other others otherwise our ours out
over own particular per perhaps rather same several shall should show shows shown since so some
such than that the their them themselves then there therefore these they this those though
three through thus to together too toward towards two under unlike until up upon us use used
uses using various very via was we well were what when where whether which while who whom whose
why will with within without would yet you your
paper papers propose proposed proposes study studies studied result results approach approaches
method methods framework new novel provide provides present presents introduce introduces
consider considers develop develops obtain obtained find finds found demonstrate demonstrates
analysis analyze analyse investigate investigates investigated work article research first second
furthermore moreover addition additionally particular specific general different important
significant significantly existing recent recently well-known key main paper's also first-order
i.e e.g et al thereby respectively allows allow allowing case cases able better large small high
low number set sets term terms way ways problem problems question questions model models modeling
modelling data i.e. e.g. can't does not tell get gets keep keeps take takes go goes going lot lots
people thing things everything something anything want wants say says said happen happens
""".split())

AIM_CUES = re.compile(
    r"\b(we (?:study|propose|investigate|examine|analy[sz]e|develop|introduce|consider|present|"
    r"explore|address|derive|construct|characteri[sz]e|test|evaluate|model|estimate|build|design)|"
    r"this (?:paper|article|study|work|note|report)|in this (?:paper|article|study|work)|"
    r"the (?:aim|goal|purpose|objective) of)\b",
    re.I,
)
CAVEAT_CUES = re.compile(
    r"\b(limitations?|caveats?|before (?:transaction )?costs|does not (?:account|capture|consider)|"
    r"do not (?:account|capture|consider)|fails? to|remains? (?:open|unclear)|future work|"
    r"in-sample only|not (?:robust|significant))\b",
    re.I,
)
TRADING_INTENT = re.compile(
    r"\b(trad\w*|strateg\w*|signals?|alpha|returns?|portfolio\w*|invest\w*|backtest\w*|hedg\w*|"
    r"buy\w*|sell\w*|profit\w*|predict\w*|forecast\w*|arbitrage|pric\w*|direction\w*|heading|"
    r"moves?|rall\w*|crash\w*)\b",
    re.I,
)
FINDING_CUES = re.compile(
    r"\b(we (?:find|show|demonstrate|document|observe|prove|establish|confirm|report)|"
    r"(?:results|findings|experiments|evidence|analysis|simulations?|estimates) (?:show|suggest|"
    r"indicate|reveal|demonstrate|confirm)|(?:outperform|improve)s?\b|predicts?\b)",
    re.I,
)

# label -> regex. The matched phrase is shown to the user as evidence.
METHOD_LEXICON: dict[str, str] = {
    "Deep learning / neural networks": r"deep learning|neural networks?|lstm|transformers?\b|convolutional",
    "Reinforcement learning": r"reinforcement learning",
    "Large language models": r"large language models?|\bllms?\b",
    "Machine learning (general)": r"machine learning|random forests?|gradient[- ]boost\w*|xgboost|lightgbm",
    "Stochastic / rough volatility models": r"stochastic volatility|rough volatility|\bheston\b|\bsabr\b|jump[- ]diffusion|l[eé]vy process",
    "GARCH / realized-volatility econometrics": r"\bgarch\b|realized volatility|\bhar model",
    "Option & derivative pricing": r"option pric\w*|black[- ]scholes|implied volatilit\w*|derivatives? pric\w*",
    "Stochastic optimal control": r"optimal control|stochastic control|hamilton[- ]jacobi|\bhjb\b|dynamic programming",
    "Monte Carlo simulation": r"monte carlo",
    "Risk measures (VaR / ES)": r"value[- ]at[- ]risk|expected shortfall|\bcvar\b|risk measures?",
    "Copulas / dependence modelling": r"copulas?",
    "Agent-based / game-theoretic": r"agent[- ]based|game[- ]theor\w*|nash equilibri\w*|mean[- ]field games?",
    "Order book / market microstructure": r"order[- ]books?|order flows?|market microstructure|limit orders?|price impact|bid[- ]ask|market[- ]mak\w+|double[- ]auction",
    "Point processes (Hawkes)": r"hawkes|point process\w*",
    "Change-point / regime detection": r"change[- ]?point|structural breaks?",
    "Portfolio optimisation": r"portfolio (?:optimi[sz]ation|selection|allocation|choice)|mean[- ]variance|markowitz",
    "Hedging": r"\bhedg\w*",
    "Tail risk / extreme events": r"tail risks?|extreme value|heavy[- ]tail\w*|fat[- ]tail\w*|crash\w*|drawdowns?",
    "Regime switching / Markov models": r"regime[- ]switching|hidden markov|markov chains?",
    "Bayesian inference": r"bayesian",
    "Network / graph methods": r"graph neural|network analysis|\bnetworks? of\b|\bgraphs?\b",
    "Regression / econometrics": r"panel data|regressions?|fama[- ]macbeth|instrumental variables?",
}

# What kind of evidence the abstract describes.
EVIDENCE_LEXICON: dict[str, str] = {
    "Empirical (real data)": r"\bempirical\w*|\bdata ?sets?\b|\bdata from\b|using [\w\- ]{0,40}\bdata\b|"
                             r"high[- ]frequency data|tick data|call report|\bcrsp\b|compustat|"
                             r"\bcase study\b|historical (?:data|events?)|\bbacktest\w*|out[- ]of[- ]sample|"
                             r"(?:options|market|price|trade|transaction|tick|daily|intraday|panel|"
                             r"real[- ]world|filings?) data\b",
    "Theoretical / mathematical": r"\bwe prove\b|\btheorems?\b|closed[- ]form|explicit (?:formula|solution)\w*|"
                                  r"\bequilibri\w+|optimality|existence and uniqueness|\baxiom\w*|\bcharacteri[sz]e\w*",
    "Simulation / numerical": r"\bsimulat\w+|numerical (?:experiments?|results|study|examples?|illustration)|"
                              r"monte carlo|synthetic (?:data|dataset)",
}

# Which markets / asset classes the abstract mentions.
MARKET_LEXICON: dict[str, str] = {
    "Equities": r"\bequit(?:y|ies)\b|\bstocks?\b|s&p|\bspx\b|nasdaq|\bcrsp\b|\bnyse\b",
    "Options / volatility": r"\boptions? (?:pric\w+|markets?|data|prices)|implied volatilit\w+|\bswaptions?\b|"
                            r"\bput options?\b|\bcall options?\b|\bvix\b|variance risk premium",
    "Futures": r"\bfutures\b|perpetual",
    "Crypto / DeFi": r"crypto\w*|bitcoin|ethereum|blockchain|\bdefi\b|stablecoins?|on-chain|\btokens?\b|uniswap",
    "FX / currencies": r"foreign exchange|\bfx\b|exchange rates?|\bcurrenc(?:y|ies)\b",
    "Rates / bonds / credit": r"\bbonds?\b|interest rates?|yield curve|treasur(?:y|ies)|\bcredit\b|\bsofr\b|"
                              r"\bloans?\b|mortgages?|\bdefault\b",
    "Commodities / energy": r"commodit(?:y|ies)|electricity|\benergy\b|\boil\b|natural gas|power markets?",
    "Insurance": r"insur\w+|reinsur\w+|catastrophe|cat bonds?|annuit\w+",
    "Banking / deposits": r"\bbanks?\b|\bdeposit\w*|bank runs?",
    "Prediction markets": r"prediction markets?|polymarket|kalshi",
}

# Practical issues a quant researcher checks before trusting a result.
PRACTICAL_CHECKS: dict[str, str] = {
    "transaction costs / frictions": r"transaction costs?|trading costs?|slippage|\bfees\b|frictions?|bid[- ]ask spread",
    "out-of-sample or backtest evaluation": r"out[- ]of[- ]sample|backtest\w*|held[- ]out|walk[- ]forward|hold[- ]out",
    "risk / drawdown reporting": r"drawdowns?|tail risk|value[- ]at[- ]risk|expected shortfall|sharpe|volatility of returns",
}

CATEGORY_NAMES = {
    "q-fin.TR": "Trading & Microstructure", "q-fin.PM": "Portfolio Management",
    "q-fin.RM": "Risk Management", "q-fin.ST": "Statistical Finance",
    "q-fin.PR": "Pricing of Securities", "q-fin.CP": "Computational Finance",
    "q-fin.MF": "Mathematical Finance", "q-fin.GN": "General Finance",
    "q-fin.EC": "Economics", "user-note": "User note",
}

_ADVERB = re.compile(r"(?:ally|ely|ily|ously|ively|ngly|ntly|ctly|rly|dly|tly)$")
_YEAR_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*(?:–|—|-|to|through|until)\s*((?:19|20)\d{2})\b")


def _compile(lexicon: dict[str, str]) -> dict[str, re.Pattern]:
    return {label: re.compile(pattern, re.I) for label, pattern in lexicon.items()}


_METHOD_RES = _compile(METHOD_LEXICON)
_EVIDENCE_RES = _compile(EVIDENCE_LEXICON)
_MARKET_RES = _compile(MARKET_LEXICON)
_PRACTICAL_RES = _compile(PRACTICAL_CHECKS)

# Similarity bands, calibrated on this corpus: for sample queries the median
# similarity of an arbitrary paper was 0.07-0.25 and on-topic matches were
# typically 0.40-0.63 (see README, "Reading similarity scores").
MATCH_BANDS = [(0.50, "Strong"), (0.40, "Good"), (0.30, "Partial"), (-1.0, "Weak")]
SCORE_HELP = ("Similarity = 1 − cosine distance between the embeddings of your text and the paper "
              "(title + abstract). In this corpus unrelated papers usually score about 0.1–0.25; "
              "≥0.50 Strong · 0.40–0.50 Good · 0.30–0.40 Partial · <0.30 Weak.")


def match_label(similarity: float) -> str:
    return next(label for threshold, label in MATCH_BANDS if similarity >= threshold)


def category_name(code: str) -> str:
    return CATEGORY_NAMES.get(code, code or "—")


# ------------------------------------------------------------------ text utils
def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(e\.g|i\.e|et al|etc|cf|vs)\.", lambda m: m.group(0).replace(".", "§"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'$\\])", text)
    return [p.replace("§", ".").strip() for p in parts if len(p.strip()) > 20]


def _normalize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith(("shes", "ches", "xes", "sses")):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def extract_terms(text: str) -> tuple[Counter, dict[str, str]]:
    """Unigram + bigram counts (stopwords removed) and a normalized->surface map."""
    # bigrams never span punctuation or line breaks
    tokens: list[str | None] = []
    for clause in re.split(r"[.,;:!?()\[\]\"\n]+", text.lower()):
        tokens.extend(re.findall(r"[a-z][a-z0-9\-]*", clause))
        tokens.append(None)
    counts: Counter = Counter()
    surface: dict[str, str] = {}
    prev: tuple[str, str] | None = None
    for tok in tokens:
        if tok is None:
            prev = None
            continue
        tok = tok.strip("-")
        if tok in STOPWORDS or len(tok) < 3 or tok.isdigit() or _ADVERB.search(tok):
            prev = None
            continue
        norm = _normalize(tok)
        counts[norm] += 1
        surface.setdefault(norm, tok)
        if prev:
            bigram = f"{prev[0]} {norm}"
            counts[bigram] += 1
            surface.setdefault(bigram, f"{prev[1]} {tok}")
        prev = (norm, tok)
    return counts, surface


_SUFFIXES = ("ations", "ation", "ings", "ing", "able", "ers", "er", "ies", "ied", "ed", "es", "s", "ly")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def word_used_in(word: str, text: str) -> bool:
    """Generous check: does any word in text share this word's stem (prefix match)?"""
    stem = _stem(word.lower())
    return any(_stem(tok) == stem or (len(stem) >= 4 and tok.startswith(stem))
               for tok in re.findall(r"[a-z]+", text.lower()))


def content_words(text: str) -> list[str]:
    """Meaningful single words of a query (used to flag very broad queries)."""
    terms, surface = extract_terms(text)
    return [surface[t] for t in terms if " " not in t]


def _tag(text: str, regexes: dict[str, re.Pattern]) -> dict[str, str]:
    """Label -> first matching phrase found in the text."""
    found = {}
    for label, regex in regexes.items():
        m = regex.search(text)
        if m:
            found[label] = m.group(0)
    return found


def detect_methods(text: str) -> dict[str, str]:
    return _tag(text, _METHOD_RES)


def detect_evidence(text: str) -> dict[str, str]:
    return _tag(text, _EVIDENCE_RES)


def detect_markets(text: str) -> dict[str, str]:
    return _tag(text, _MARKET_RES)


def detect_practical(text: str) -> dict[str, str]:
    return _tag(text, _PRACTICAL_RES)


def data_period(text: str) -> str:
    m = _YEAR_RANGE.search(text)
    return f"{m.group(1)}–{m.group(2)}" if m and m.group(1) < m.group(2) else ""


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def short_authors(authors: str, limit: int = 3) -> str:
    names = [a.strip() for a in authors.split(",") if a.strip()]
    if len(names) <= limit:
        return ", ".join(names) or "Unknown"
    return f"{', '.join(names[:limit])}, et al."


def citation(r: SearchResult) -> str:
    if r.is_note:
        src = f" {r.url}" if r.url else ""
        return f"{r.authors} ({r.year}). {r.title}. User-added note.{src}"
    arxiv_id = r.metadata.get("arxiv_id", "")
    return f"{short_authors(r.authors)} ({r.year}). *{r.title}*. arXiv:{arxiv_id}. {r.url}"


def query_feedback(query: str, results: list[SearchResult]) -> list[str]:
    """Plain-language tips when a query is very broad or nothing matches well."""
    tips = []
    if len(re.findall(r"[A-Za-z]{2,}", query)) < 4 or not content_words(query):
        tips.append("Your query is very short or broad, so results may cover unrelated sub-topics. "
                    "Try describing the mechanism, market and outcome you care about, e.g. "
                    "\"Does heavy option buying before earnings predict stock returns?\" "
                    "Single technical terms give the embedding model little context.")
    if results and results[0].similarity < 0.30:
        tips.append(f"The best match is weak (similarity {results[0].similarity:.2f}). The corpus may "
                    f"not cover this topic directly; treat these as adjacent work or rephrase.")
    return tips


# --------------------------------------------------------------- paper digest
@dataclass
class PaperDigest:
    result: SearchResult
    aim: str
    aim_is_explicit: bool
    findings: list[str]
    caveats: list[str]
    best_passage: str
    best_passage_similarity: float
    methods: dict[str, str]
    evidence: dict[str, str]
    markets: dict[str, str]
    practical: dict[str, str]
    period: str
    terms: Counter
    surface: dict[str, str]
    title_terms: set[str] = field(default_factory=set)
    embedding: list[float] = field(default_factory=list)


def digest_papers(db: ResearchDatabase, query: str, results: list[SearchResult]) -> list[PaperDigest]:
    """Extract grounded facts from each abstract and the passage closest to the query."""
    all_sentences: list[list[str]] = [split_sentences(r.abstract) or [r.abstract] for r in results]
    flat = [s for group in all_sentences for s in group]
    vectors = db.embed([query] + flat + [r.document for r in results])
    q_vec, rest = vectors[0], vectors[1:]
    sent_vecs, doc_vecs = rest[:len(flat)], rest[len(flat):]

    digests, offset = [], 0
    for r, sentences, doc_vec in zip(results, all_sentences, doc_vecs):
        sims = [cosine(q_vec, v) for v in sent_vecs[offset:offset + len(sentences)]]
        offset += len(sentences)
        best = max(range(len(sentences)), key=lambda i: sims[i])
        aim = next((s for s in sentences if AIM_CUES.search(s)), None)
        findings = [s for s in sentences if FINDING_CUES.search(s) and s != aim][:2]
        terms, surface = extract_terms(r.document)
        digests.append(PaperDigest(
            result=r,
            aim=aim or sentences[0],
            aim_is_explicit=aim is not None,
            findings=findings,
            caveats=[s for s in sentences if CAVEAT_CUES.search(s)][:1],
            best_passage=sentences[best],
            best_passage_similarity=sims[best],
            methods=detect_methods(r.document),
            evidence=detect_evidence(r.document),
            markets=detect_markets(r.document),
            practical=detect_practical(r.document),
            period=data_period(r.abstract),
            terms=terms,
            surface=surface,
            title_terms=set(extract_terms(r.title)[0]),
            embedding=doc_vec,
        ))
    return digests


def _rank_terms(digests: list[PaperDigest]) -> tuple[list[tuple[str, list[int]]], dict[int, list[str]]]:
    """Shared terms (in >=2 abstracts) and distinctive terms per paper."""
    df: Counter = Counter()
    for d in digests:
        df.update(set(d.terms))
    surface = {}
    for d in digests:
        for k, v in d.surface.items():
            surface.setdefault(k, v)

    def weight(term: str) -> float:
        return 1.6 if " " in term else 1.0

    shared_terms = [t for t, n in df.items() if n >= 2]
    shared_terms.sort(key=lambda t: (-df[t] * weight(t), -sum(d.terms[t] for d in digests), t))
    shared = []
    for t in shared_terms:
        # skip a unigram already covered by a chosen bigram
        if any(t in s.split() for s, _ in shared if " " in s):
            continue
        shared.append((surface[t], [i for i, d in enumerate(digests) if t in d.terms]))
        if len(shared) == 8:
            break

    distinctive = {}
    for i, d in enumerate(digests):
        own = [t for t in d.terms if df[t] == 1]
        # prefer terms that recur in the abstract or appear in the title
        own.sort(key=lambda t: (-(d.terms[t] + 2 * (t in d.title_terms) + 0.5 * (" " in t)), t))
        picked: list[str] = []
        for t in own:
            if any(t in p or p in t for p in picked):
                continue
            picked.append(t)
            if len(picked) == 5:
                break
        distinctive[i] = [surface[t] for t in picked]
    return shared, distinctive


def _group(digests: list[PaperDigest], attr: str) -> dict[str, list[int]]:
    """Tag label -> indices of papers carrying it (e.g. attr='markets')."""
    out: dict[str, list[int]] = {}
    for i, d in enumerate(digests):
        for label in getattr(d, attr):
            out.setdefault(label, []).append(i)
    return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0])))


# ---------------------------------------------------------------------- brief
@dataclass
class ResearchBrief:
    question: str
    digests: list[PaperDigest]
    confidence: str
    confidence_note: str
    shared_terms: list[tuple[str, list[int]]]
    distinctive_terms: dict[int, list[str]]
    methods: dict[str, list[int]]
    evidence: dict[str, list[int]]
    markets: dict[str, list[int]]
    practical: dict[str, list[int]]
    closest_pair: tuple[int, int, float] | None
    most_distinct: tuple[int, float] | None
    unused_query_words: list[str]
    search_terms: list[str]
    gaps: list[str]
    follow_ups: list[str]
    query_tips: list[str]
    generated_at: datetime
    trading_question: bool = False

    @property
    def shared_methods(self) -> dict[str, list[int]]:
        return {m: idx for m, idx in self.methods.items() if len(idx) >= 2}


def _confidence(digests: list[PaperDigest]) -> tuple[str, str]:
    sims = [d.result.similarity for d in digests]
    good = sum(s >= 0.40 for s in sims)
    if sims[0] >= 0.50 and good >= 2:
        return "High", f"The best match scores {sims[0]:.2f} and {good} of {len(sims)} papers score ≥ 0.40."
    if sims[0] >= 0.40:
        return "Moderate", (f"The best match scores {sims[0]:.2f}; {good} of {len(sims)} papers score "
                            f"≥ 0.40. Expect related rather than exact-match literature.")
    return "Low", (f"The best match only scores {sims[0]:.2f}. The corpus may not contain papers that "
                   f"address this question directly; treat these as adjacent work.")


def generate_brief(db: ResearchDatabase, question: str, k: int = 5) -> ResearchBrief | None:
    """Build a structured brief from the top-k (3-5) arXiv papers for the question."""
    k = max(3, min(k, 5))
    results = db.search(question, k=k, source_type=SOURCE_ARXIV)
    if not results:
        return None
    digests = digest_papers(db, question, results)
    shared_terms, distinctive = _rank_terms(digests)
    confidence, confidence_note = _confidence(digests)

    closest_pair = most_distinct = None
    if len(digests) >= 2:
        pairs = [(i, j, cosine(digests[i].embedding, digests[j].embedding))
                 for i in range(len(digests)) for j in range(i + 1, len(digests))]
        closest_pair = max(pairs, key=lambda p: p[2])
        if len(digests) >= 3:
            avg = [(i, sum(cosine(digests[i].embedding, digests[j].embedding)
                           for j in range(len(digests)) if j != i) / (len(digests) - 1))
                   for i in range(len(digests))]
            most_distinct = min(avg, key=lambda a: a[1])

    all_text = " ".join(d.result.document for d in digests)
    unused = [w for w in content_words(question) if not word_used_in(w, all_text)]
    search_terms = literature_vocabulary(question, digests, shared_terms)

    brief = ResearchBrief(
        question=question,
        digests=digests,
        confidence=confidence,
        confidence_note=confidence_note,
        shared_terms=shared_terms,
        distinctive_terms=distinctive,
        methods=_group(digests, "methods"),
        evidence=_group(digests, "evidence"),
        markets=_group(digests, "markets"),
        practical=_group(digests, "practical"),
        closest_pair=closest_pair,
        most_distinct=most_distinct,
        unused_query_words=unused,
        search_terms=search_terms,
        gaps=[],
        follow_ups=[],
        query_tips=query_feedback(question, results),
        generated_at=datetime.now(),
        trading_question=bool(TRADING_INTENT.search(question)),
    )
    brief.gaps = find_gaps(brief)
    brief.follow_ups = suggest_follow_ups(brief)
    return brief


def literature_vocabulary(question: str, digests: list[PaperDigest],
                          shared_terms: list[tuple[str, list[int]]], limit: int = 5) -> list[str]:
    """Phrases the papers use for this topic that the question does not: terms shared by
    several abstracts first, then multi-word phrases from the best-matching titles."""
    out: list[str] = []

    def add(term: str) -> None:
        if (len(out) < limit and term not in out and not any(term in o or o in term for o in out)
                and not all(word_used_in(w, question) for w in term.split())):
            out.append(term)

    for term, _ in shared_terms:
        if " " in term:
            add(term)
    title_words: set[str] = set()
    for d in sorted(digests, key=lambda d: -d.result.similarity)[:2]:
        title_terms, surface = extract_terms(d.result.title)
        # a title phrase that recurs in the abstract is a genuine key phrase
        for t in title_terms:
            if " " in t and d.terms[t] >= 2 and not set(t.split()) & title_words:
                before = len(out)
                add(surface[t])
                if len(out) > before:
                    title_words.update(t.split())
    return out


def _refs(idx: list[int]) -> str:
    return ", ".join(f"[{i + 1}]" for i in idx)


def find_gaps(brief: ResearchBrief) -> list[str]:
    """Things the retrieved abstracts do not mention. Phrased as observations about abstracts."""
    n, gaps = len(brief.digests), []
    for i, d in enumerate(brief.digests, 1):
        for c in d.caveats:
            gaps.append(f"Caveat stated by the authors of [{i}]: \"{c}\"")
    if brief.trading_question:
        for check in PRACTICAL_CHECKS:
            if check not in brief.practical:
                gaps.append(f"None of the {n} abstracts mention **{check}**.")
    for evidence in EVIDENCE_LEXICON:
        if evidence not in brief.evidence:
            gaps.append(f"No abstract describes **{evidence.lower()}** work.")
    if brief.markets:
        top_market, idx = next(iter(brief.markets.items()))
        if len(idx) == n and len(brief.markets) == 1:
            gaps.append(f"All papers concern **{top_market}**; nothing here tests the idea in "
                        f"other markets.")
    else:
        gaps.append("No abstract names a specific market or asset class.")
    if not any(d.period for d in brief.digests):
        gaps.append("No abstract states the time period of its data.")
    if brief.confidence == "Low":
        gaps.append("Retrieval confidence is low, so the stored corpus may simply lack papers on "
                    "this exact question.")
    return gaps


def suggest_follow_ups(brief: ResearchBrief) -> list[str]:
    """Exactly three practical follow-up questions, prioritised by the gaps found."""
    ds = brief.digests
    candidates: list[str] = []

    # anchor on the most relevant paper that reports empirical work (else the top paper)
    empirical = brief.evidence.get("Empirical (real data)", [])
    anchor = empirical[0] if empirical else 0
    a_ref = f"[{anchor + 1}]"

    if brief.trading_question and "transaction costs / frictions" not in brief.practical:
        candidates.append(f"Does the effect described in {a_ref} survive realistic transaction costs, "
                          f"slippage and capacity limits? None of the abstracts mention costs.")
    if brief.trading_question and "out-of-sample or backtest evaluation" not in brief.practical:
        candidates.append(f"Does the result in {a_ref} hold out-of-sample, e.g. on a later time period "
                          f"or a walk-forward test?")
    # market transfer: a paper studied in one market, another market present in the set
    for i, d in enumerate(ds):
        own = list(d.markets)
        others = [m for m in brief.markets if m not in d.markets]
        if own and others:
            candidates.append(f"Would the approach of [{i + 1}] (studied on {own[0]}) carry "
                              f"over to {others[0]}, which [{brief.markets[others[0]][0] + 1}] "
                              f"examines?")
            break
    # method transfer between papers
    method_sets = [set(d.methods) for d in ds]
    for i in range(len(ds)):
        j = next((j for j in range(len(ds)) if j != i and method_sets[i] - method_sets[j]
                  and brief.distinctive_terms.get(j)), None)
        if j is not None:
            m = sorted(method_sets[i] - method_sets[j])[0]
            title = ds[j].result.title
            title = title if len(title) <= 70 else title[:67].rsplit(" ", 1)[0] + "…"
            candidates.append(f"Could the {m.lower()} approach of [{i + 1}] be applied to the problem "
                              f"studied in [{j + 1}] (*{title}*)?")
            break
    shared = [(t, idx) for t, idx in brief.shared_terms if " " in t] or brief.shared_terms
    if shared:
        term, idx = shared[0]
        candidates.append(f"Do the papers that discuss \"{term}\" ({_refs(idx)}) reach compatible "
                          f"conclusions when compared on the same data and period?")
    candidates.append(f"What would a minimal replication of [{anchor + 1}] require (data, horizon, "
                      f"benchmark), and is that data available to you?")
    return candidates[:3]


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return " ".join(text.replace("|", "\\|").split())


def _short_labels(labels, limit: int = 3) -> str:
    labels = list(labels)
    if not labels:
        return "—"
    shown = ", ".join(labels[:limit])
    return shown + (f" +{len(labels) - limit}" if len(labels) > limit else "")


def _short_evidence(label: str) -> str:
    return label.split(" (")[0].split(" /")[0]


def brief_to_markdown(brief: ResearchBrief) -> str:
    ds, n = brief.digests, len(brief.digests)
    lines = [
        "# AlphaSearch Research Brief",
        "",
        f"**Question:** {brief.question}  ",
        f"**Generated:** {brief.generated_at:%Y-%m-%d %H:%M} · **Papers analysed:** {n} arXiv abstracts · "
        f"**Retrieval confidence:** {brief.confidence}",
        "",
        "## At a glance",
        "",
        f"- **Retrieval confidence: {brief.confidence}.** {brief.confidence_note}",
    ]
    if brief.methods:
        top = [(m, idx) for m, idx in brief.methods.items() if len(idx) >= 2][:2]
        if top:
            lines.append("- **Recurring themes:** " + "; ".join(
                f"{m} ({len(idx)} of {n})" for m, idx in top) + ".")
    if brief.evidence:
        lines.append("- **Evidence base:** " + "; ".join(
            f"{_short_evidence(e).lower()} in {_refs(idx)}" for e, idx in brief.evidence.items()) + ".")
    if brief.markets:
        lines.append("- **Markets studied:** " + "; ".join(
            f"{m} {_refs(idx)}" for m, idx in list(brief.markets.items())[:4]) + ".")
    best_passage = max(range(n), key=lambda i: ds[i].best_passage_similarity)
    lines.append(f"- **Start with:** [1] (highest overall similarity)"
                 + (f" and [{best_passage + 1}] (its abstract contains the sentence closest to your "
                    f"question)." if best_passage != 0 else "."))
    if brief.search_terms:
        lines.append("- **Search with the literature's vocabulary:** "
                     + ", ".join(f"\"{t}\"" for t in brief.search_terms) + ".")
    for tip in brief.query_tips:
        lines.append(f"- **Note:** {tip}")

    lines += ["", "## 1. Research question", "", f"> {brief.question}", "",
              "## 2. Most relevant papers", "",
              "| # | Paper | Year | Category | Similarity | Match |",
              "|---|---|---|---|---|---|"]
    for i, d in enumerate(ds, 1):
        r = d.result
        lines.append(f"| {i} | [{_cell(r.title)}]({r.url}) | {r.year} | "
                     f"{_cell(category_name(r.metadata.get('category', '')))} | "
                     f"{r.similarity:.3f} | {match_label(r.similarity)} |")
    lines += ["", f"*{SCORE_HELP}*", ""]

    lines += ["## 3. What each paper investigates", "",
              "*Quoted sentences are verbatim from each abstract. Tags show the abstract phrase that "
              "triggered them.*", ""]
    for i, d in enumerate(ds, 1):
        r = d.result
        lines += [f"### [{i}] {r.title}", "",
                  f"*{short_authors(r.authors)} · {r.year} · {category_name(r.metadata.get('category', ''))}*",
                  ""]
        label = "Aim" if d.aim_is_explicit else "Opening sentence (no explicit aim statement found)"
        lines.append(f"- **{label}:** \"{d.aim}\"")
        if d.findings:
            for f in d.findings:
                lines.append(f"- **Reported result:** \"{f}\"")
        else:
            lines.append("- **Reported result:** no explicit result statement in the abstract.")
        for c in d.caveats:
            if c not in d.findings:
                lines.append(f"- **Stated caveat:** \"{c}\"")
        if d.evidence:
            lines.append("- **Evidence type:** " + "; ".join(
                f"{e} (\"{p}\")" for e, p in d.evidence.items()))
        if d.markets:
            lines.append("- **Markets:** " + "; ".join(f"{m} (\"{p}\")" for m, p in d.markets.items()))
        if d.period:
            lines.append(f"- **Data period mentioned:** {d.period}")
        if d.methods:
            lines.append("- **Methods:** " + "; ".join(f"{m} (\"{p}\")" for m, p in d.methods.items()))
        lines.append("")

    lines += ["## 4. How the papers connect to your question", "",
              "For each paper, the abstract sentence whose meaning is closest to your question:", ""]
    for i, d in enumerate(ds, 1):
        lines.append(f"- **[{i}]** ({match_label(d.result.similarity).lower()} match, sentence similarity "
                     f"{d.best_passage_similarity:.2f}) \"{d.best_passage}\"")
    lines.append("")
    if brief.unused_query_words:
        words = ", ".join(f"\"{w}\"" for w in brief.unused_query_words[:6])
        lines.append(f"Found by meaning, not wording: no retrieved title or abstract contains "
                     f"{words} from your question (or another form of those words).")
        lines.append("")

    lines += ["## 5. Similarities and differences", "",
              "| # | Evidence | Markets | Main methods |", "|---|---|---|---|"]
    for i, d in enumerate(ds, 1):
        evidence = _short_labels(_short_evidence(e) for e in d.evidence).replace("—", "not stated")
        lines.append(f"| {i} | {_cell(evidence)} | "
                     f"{_cell(_short_labels(d.markets))} | {_cell(_short_labels(d.methods, 2))} |")
    lines.append("")
    common = [f"{m} ({_refs(idx)})" for m, idx in brief.methods.items() if len(idx) >= 2]
    common += [f"{m} ({_refs(idx)})" for m, idx in brief.markets.items() if len(idx) >= 2]
    lines.append("**What they share (heuristic):** " + ("; ".join(common) + "." if common else
                 "no method or market is shared by two or more papers. They approach the question "
                 "from separate directions."))
    if brief.shared_terms:
        lines.append("Common vocabulary: " + ", ".join(
            f"\"{t}\" {_refs(idx)}" for t, idx in brief.shared_terms[:6]) + ".")
    lines += ["", "**Where they differ (heuristic):**", ""]
    for i, d in enumerate(ds):
        terms = ", ".join(brief.distinctive_terms.get(i, [])[:4]) or "—"
        lines.append(f"- [{i + 1}] distinctive terms: {terms}")
    if brief.closest_pair or brief.most_distinct:
        lines.append("")
    if brief.closest_pair:
        i, j, s = brief.closest_pair
        lines.append(f"[{i + 1}] and [{j + 1}] are the most alike (abstract similarity {s:.2f}). ")
    if brief.most_distinct:
        i, s = brief.most_distinct
        lines.append(f"[{i + 1}] is the least like the others (average similarity {s:.2f}), so it "
                     f"likely offers a different angle.")
    lines.append("")

    lines += ["## 6. Gaps and open questions", "",
              "*Based only on what the abstracts state. A gap here means \"not mentioned in the "
              "abstract\", not \"not done in the paper\".*", ""]
    lines += [f"- {g}" for g in brief.gaps] or ["- No obvious gaps were detected from the abstracts."]
    lines.append("")

    lines += ["## 7. Follow-up research questions", ""]
    lines += [f"{i}. {q}" for i, q in enumerate(brief.follow_ups, 1)]
    lines.append("")

    lines += ["## 8. References", ""]
    lines += [f"{i}. {citation(d.result)}" for i, d in enumerate(ds, 1)]
    lines += ["", "---", "",
              "*How this brief was made: papers were retrieved by semantic search over a local "
              "ChromaDB collection (all-MiniLM-L6-v2 embeddings, cosine similarity). No language "
              "model was used. Quotes are verbatim; tags come from fixed phrase lists; comparisons "
              "and gaps are heuristics computed from the abstracts only. Read the full papers before "
              "relying on any conclusion.*", ""]
    return "\n".join(lines)


def export_brief(markdown: str, question: str, out_dir: Path = OUTPUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:50] or "brief"
    path = out_dir / f"brief_{datetime.now():%Y%m%d_%H%M%S}_{slug}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


# ------------------------------------------------------------ idea assessment
# Same thresholds as MATCH_BANDS so the idea rating and the per-paper labels agree.
COVERAGE_BANDS = [
    (0.50, "Closely studied", "Papers in the corpus address something very similar. Read these first "
                              "and look for what they do NOT cover."),
    (0.40, "Related work exists", "There is overlapping literature, but not an exact match. Your idea "
                                  "may combine or extend these approaches."),
    (0.30, "Loosely related", "Only partially related work was found in this corpus. Your idea may be "
                              "less explored here, or phrased very differently."),
    (-1.0, "Little related work in this corpus", "Nothing close was found among the stored papers. "
                                                 "Try rephrasing, or import more papers."),
]


@dataclass
class IdeaAssessment:
    idea: str
    digests: list[PaperDigest]
    band: str
    band_advice: str
    shared_terms: dict[int, list[str]]
    uncovered_idea_terms: list[str]
    methods_to_explore: dict[str, list[int]]
    markets: dict[str, list[int]]
    evidence: dict[str, list[int]]
    search_terms: list[str]
    tips: list[str]


def assess_idea(db: ResearchDatabase, idea: str, k: int = 5) -> IdeaAssessment | None:
    """Compare a user's research idea against the most similar stored papers."""
    results = db.search(idea, k=k, source_type=SOURCE_ARXIV)
    if not results:
        return None
    digests = digest_papers(db, idea, results)
    idea_terms, idea_surface = extract_terms(idea)
    top = results[0].similarity
    band, advice = next((b, a) for threshold, b, a in COVERAGE_BANDS if top >= threshold)

    shared = {}
    for i, d in enumerate(digests):
        common = [idea_surface[t] for t in idea_terms if t in d.terms]
        common = [t for t in common if not any(t != o and t in o for o in common)]
        shared[i] = common[:6]
    all_text = " ".join(d.result.document for d in digests)
    uncovered = [w for w in content_words(idea) if not word_used_in(w, all_text)]
    shared_terms, _ = _rank_terms(digests)
    search_terms = literature_vocabulary(idea, digests, shared_terms)

    return IdeaAssessment(idea, digests, band, advice, shared, uncovered,
                          _group(digests, "methods"), _group(digests, "markets"),
                          _group(digests, "evidence"), search_terms,
                          query_feedback(idea, results))

"""Shared fixtures. Every test uses a temporary ChromaDB directory, never ./chroma_db.

The FIXTURE_PAPERS below are short synthetic texts written for testing only.
They are not real papers and are never written to the real database.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import ResearchDatabase  # noqa: E402

FIXTURE_PAPERS = [
    {
        "arxiv_id": "test.0001",
        "title": "Tail-risk hedging with out-of-the-money put options",
        "abstract": "We study portfolio insurance strategies that buy deep out-of-the-money put "
                    "options to limit drawdowns during equity crashes. We find that systematic "
                    "put protection reduces maximum drawdown but carries a persistent premium cost.",
        "authors": ["Test Author A"],
        "published": "2024-01-15",
        "primary_category": "q-fin.RM",
        "categories": ["q-fin.RM"],
        "url": "https://example.org/test.0001",
    },
    {
        "arxiv_id": "test.0002",
        "title": "Order-flow imbalance and short-horizon price impact",
        "abstract": "This paper examines how imbalances between buyer- and seller-initiated limit "
                    "order book events forecast mid-price changes over seconds. Results show that "
                    "order-flow imbalance explains a large share of short-term price impact.",
        "authors": ["Test Author B", "Test Author C"],
        "published": "2023-06-01",
        "primary_category": "q-fin.TR",
        "categories": ["q-fin.TR", "q-fin.ST"],
        "url": "https://example.org/test.0002",
    },
    {
        "arxiv_id": "test.0003",
        "title": "Calibrating stochastic volatility models to implied volatility surfaces",
        "abstract": "We propose a fast calibration of the Heston stochastic volatility model to "
                    "listed option prices using neural network approximations of the pricing map. "
                    "Experiments show calibration errors comparable to Monte Carlo benchmarks.",
        "authors": ["Test Author D"],
        "published": "2022-11-30",
        "primary_category": "q-fin.CP",
        "categories": ["q-fin.CP", "q-fin.PR"],
        "url": "https://example.org/test.0003",
    },
    {
        "arxiv_id": "test.0004",
        "title": "Mean-variance portfolio selection with transaction costs",
        "abstract": "We develop a multi-period Markowitz portfolio optimisation framework that "
                    "penalises turnover. We show that accounting for trading costs improves "
                    "out-of-sample Sharpe ratios for a universe of large-cap equities.",
        "authors": ["Test Author E"],
        "published": "2021-03-10",
        "primary_category": "q-fin.PM",
        "categories": ["q-fin.PM"],
        "url": "https://example.org/test.0004",
    },
    {
        "arxiv_id": "test.0005",
        "title": "Catastrophe bonds and the transfer of natural disaster risk",
        "abstract": "We analyse how insurers issue catastrophe bonds to move hurricane and "
                    "earthquake losses to capital-market investors, and we estimate the yield spread "
                    "investors require for bearing this disaster risk.",
        "authors": ["Test Author F"],
        "published": "2020-08-20",
        "primary_category": "q-fin.RM",
        "categories": ["q-fin.RM"],
        "url": "https://example.org/test.0005",
    },
]


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "chroma_test"


@pytest.fixture
def db(db_path) -> ResearchDatabase:
    return ResearchDatabase(db_path)


@pytest.fixture
def loaded_db(db) -> ResearchDatabase:
    db.upsert_papers(FIXTURE_PAPERS)
    return db

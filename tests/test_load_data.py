"""arXiv ingestion tests. Network calls are mocked; no real requests are made."""

import pytest
import requests

import load_data
from load_data import ArxivFetchError, fetch_papers, import_papers, parse_feed, save_cache

# Minimal Atom feed in arXiv's format, with synthetic content for parser testing.
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/9999.00001v2</id>
    <title>Parser   fixture
      title one</title>
    <summary>  Synthetic abstract used only to test XML parsing.  </summary>
    <published>2025-02-03T10:00:00Z</published>
    <author><name>Alice Example</name></author>
    <author><name>Bob Example</name></author>
    <arxiv:primary_category term="q-fin.TR"/>
    <category term="q-fin.TR"/><category term="q-fin.ST"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/9999.00002v1</id>
    <title>Parser fixture title two</title>
    <summary>Second synthetic abstract.</summary>
    <published>2025-02-04T10:00:00Z</published>
    <author><name>Carol Example</name></author>
    <arxiv:primary_category term="q-fin.RM"/>
    <category term="q-fin.RM"/>
  </entry>
</feed>"""


class FakeResponse:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses, self.headers, self.calls = list(responses), {}, 0

    def get(self, *args, **kwargs):
        self.calls += 1
        r = self.responses.pop(0) if self.responses else requests.ConnectionError("offline")
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(load_data.time, "sleep", lambda s: None)


def test_parse_feed():
    papers, total = parse_feed(FEED)
    assert total == 2 and len(papers) == 2
    p = papers[0]
    assert p["arxiv_id"] == "9999.00001"
    assert p["title"] == "Parser fixture title one"
    assert p["abstract"] == "Synthetic abstract used only to test XML parsing."
    assert p["authors"] == ["Alice Example", "Bob Example"]
    assert p["published"] == "2025-02-03"
    assert p["primary_category"] == "q-fin.TR"
    assert p["url"] == "https://arxiv.org/abs/9999.00001"


def test_fetch_retries_then_succeeds():
    session = FakeSession([requests.ConnectionError("blip"), FakeResponse("", 503), FakeResponse(FEED)])
    papers = fetch_papers(target=2, categories=["q-fin.TR"], session=session, progress=lambda m: None)
    assert len(papers) == 2 and session.calls == 3


def test_fetch_failure_raises_and_writes_nothing(db, tmp_path):
    cache = tmp_path / "cache.json"
    session = FakeSession([])  # every call raises ConnectionError
    monkey_fetch = lambda *a, **k: fetch_papers(*a, session=session, **k)  # noqa: E731
    original = load_data.fetch_papers
    load_data.fetch_papers = monkey_fetch
    try:
        with pytest.raises(ArxivFetchError):
            import_papers(db, target=5, categories=["q-fin.TR"], progress=lambda m: None,
                          cache_path=cache)
    finally:
        load_data.fetch_papers = original
    assert db.count() == 0
    assert not cache.exists()


def test_import_from_cache_is_idempotent(db, tmp_path):
    cache = tmp_path / "cache.json"
    papers, _ = parse_feed(FEED)
    save_cache(papers, cache)
    first = import_papers(db, from_cache=True, cache_path=cache, progress=lambda m: None)
    second = import_papers(db, from_cache=True, cache_path=cache, progress=lambda m: None)
    assert first["added"] == 2 and second["added"] == 0 and second["skipped_existing"] == 2
    assert db.count() == 2


def test_import_from_missing_cache(db, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_papers(db, from_cache=True, cache_path=tmp_path / "missing.json", progress=lambda m: None)


def test_fetch_by_ids_and_ids_file(tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# comment\n9999.00001\n\n9999.00002  # trailing comment\n", encoding="utf-8")
    ids = load_data.read_ids_file(ids_file)
    assert ids == ["9999.00001", "9999.00002"]
    session = FakeSession([FakeResponse(FEED)])
    papers = load_data.fetch_by_ids(ids, session=session, progress=lambda m: None)
    assert [p["arxiv_id"] for p in papers] == ids and session.calls == 1

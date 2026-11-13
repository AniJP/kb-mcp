"""Phase 4: ingestion is idempotent, deduplicating, safe, and reports what it did."""

import pytest

from kb_mcp import ingest
from kb_mcp.embeddings import HashingEmbedder
from kb_mcp.store import KnowledgeStore


@pytest.fixture
def store():
    s = KnowledgeStore(embedder=HashingEmbedder())
    yield s
    s.close()


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "kafka.md").write_text(
        "---\ntitle: Kafka retention\ntags: [kafka, streaming]\n---\nretention.ms deletes old segments. #ops\n"
    )
    (tmp_path / "dbt").mkdir()
    (tmp_path / "dbt" / "snapshots.md").write_text("# dbt snapshots\nType 2 history of mutable rows.\n")
    (tmp_path / "plain-note.txt").write_text("No heading here, title comes from the file name.")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "config.md").write_text("ignore me")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    return tmp_path


def titles(store):
    return sorted(n.title for n in store.list(limit=100)[0])


def test_folder_parsing_title_and_tags(vault):
    docs = {d.title: d for d in ingest.folder_documents(vault)}
    assert set(docs) == {"Kafka retention", "dbt snapshots", "plain note"}
    assert docs["Kafka retention"].tags == ["kafka", "ops", "streaming"]
    assert "# dbt snapshots" not in docs["dbt snapshots"].body


def test_folder_ingest_is_idempotent(store, vault):
    r1 = ingest.ingest_folder(store, vault)
    assert r1.counts["added"] == 3 and r1.counts["failed"] == 0
    r2 = ingest.ingest_folder(store, vault)
    assert r2.counts == {"added": 0, "updated": 0, "unchanged": 3, "linked": 0, "pruned": 0, "failed": 0}
    assert len(titles(store)) == 3


def test_changed_file_updates_same_note(store, vault):
    ingest.ingest_folder(store, vault)
    before = {n.title: n.id for n in store.list()[0]}
    (vault / "dbt" / "snapshots.md").write_text("# dbt snapshots\nNow using the check strategy.\n")
    r = ingest.ingest_folder(store, vault)
    assert r.counts["updated"] == 1 and r.counts["unchanged"] == 2
    note = store.get(before["dbt snapshots"])
    assert "check strategy" in note.body
    assert store.semantic_search("check strategy")[0].note.id == note.id  # re-embedded


def test_prune_only_removes_notes_from_that_folder(store, vault, tmp_path_factory):
    hand_written = store.add("My own note")
    other = tmp_path_factory.mktemp("other")
    (other / "keep.md").write_text("# Other vault note")
    ingest.ingest_folder(store, vault)
    ingest.ingest_folder(store, other)
    (vault / "plain-note.txt").unlink()
    assert ingest.ingest_folder(store, vault).counts["pruned"] == 0  # prune is opt-in
    r = ingest.ingest_folder(store, vault, prune=True)
    assert r.counts["pruned"] == 1
    remaining = titles(store)
    assert "plain note" not in remaining
    assert {"My own note", "Other vault note"} <= set(remaining) and store.get(hand_written.id)


def test_bad_file_does_not_stop_the_run(store, vault, monkeypatch):
    real = ingest.parse_note_file

    def flaky(path, root):
        if path.name == "kafka.md":
            raise UnicodeError("bad bytes")
        return real(path, root)

    monkeypatch.setattr(ingest, "parse_note_file", flaky)
    r = ingest.ingest_folder(store, vault)
    assert r.counts["added"] == 2 and r.counts["failed"] == 1 and "kafka.md" in r.errors[0]
    assert store.recent_runs(1)[0]["failed"] == 1


BOOKMARKS = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE><H1>Bookmarks</H1>
<DL><p>
  <DT><H3>Bookmarks bar</H3>
  <DL><p>
    <DT><A HREF="https://duckdb.org/docs/?utm_source=newsletter#intro">DuckDB docs</A>
    <DT><H3>Data Engineering</H3>
    <DL><p>
      <DT><A HREF="https://kafka.apache.org/documentation/">Kafka docs</A>
      <DT><A HREF="javascript:alert(1)">bookmarklet</A>
    </DL><p>
  </DL><p>
  <DT><A HREF="https://example.com/top-level">Top level &amp; co</A>
</DL><p>"""


def test_bookmarks_folders_become_tags_and_junk_is_skipped(tmp_path):
    f = tmp_path / "bookmarks.html"
    f.write_text(BOOKMARKS)
    docs = {d.title: d for d in ingest.bookmark_documents(f)}
    assert set(docs) == {"DuckDB docs", "Kafka docs", "Top level & co"}
    assert docs["Kafka docs"].tags == ["data-engineering"]
    assert docs["DuckDB docs"].tags == []  # browser root folders are not tags
    assert docs["DuckDB docs"].url == "https://duckdb.org/docs"


def test_bookmark_matching_a_saved_url_is_linked_not_duplicated(store, tmp_path):
    saved = store.add("DuckDB", "my own description", url="https://duckdb.org/docs")
    f = tmp_path / "bookmarks.html"
    f.write_text(BOOKMARKS)
    r = ingest.ingest_bookmarks(store, f)
    assert r.counts["linked"] == 1 and r.counts["added"] == 2
    assert store.get(saved.id).body == "my own description"  # the hand-written note is untouched
    assert ingest.ingest_bookmarks(store, f).counts["unchanged"] == 3


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("HTTPS://Example.COM/Path/?utm_campaign=x&id=3#frag", "https://example.com/Path?id=3"),
        ("https://example.com/", "https://example.com"),
        ("https://example.com/a?fbclid=1", "https://example.com/a"),
    ],
)
def test_normalize_url(raw, clean):
    assert ingest.normalize_url(raw) == clean


def test_readable_text_drops_page_chrome():
    html = ("<html><head><title> My &amp; Page </title><style>.x{}</style></head><body><nav>Menu</nav>"
            "<main><h2>Section</h2><p>" + "Real content here. " * 20 + "</p></main>"
            "<aside>Ads</aside><footer>Footer</footer></body></html>")
    title, text = ingest.extract_page(html)
    assert title == "My & Page"
    assert "## Section" in text and "Real content" in text
    assert not any(junk in text for junk in ("Menu", "Ads", "Footer", ".x{}"))


@pytest.mark.parametrize(
    ("url", "addresses"),
    [
        ("http://localhost/admin", ["127.0.0.1"]),
        ("http://metadata.internal/", ["169.254.169.254"]),
        ("http://router/", ["192.168.1.1"]),
        ("http://v6/", ["::1"]),
        ("ftp://example.com/x", ["93.184.216.34"]),
    ],
)
def test_ssrf_guard_blocks_non_public_targets(url, addresses):
    with pytest.raises(ingest.UnsafeURL):
        ingest.check_public_url(url, resolve=lambda host: addresses)


def test_ssrf_guard_allows_public_hosts():
    ingest.check_public_url("https://example.com/a", resolve=lambda host: ["93.184.216.34"])


def test_failed_fetch_is_reported_not_raised(store):
    def boom(url):
        raise TimeoutError("timed out")

    r = ingest.ingest_url(store, "https://example.com/slow", fetch=boom)
    assert r.counts["failed"] == 1 and "timed out" in r.errors[0]
    assert store.list()[1] == 0

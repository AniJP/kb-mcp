"""Storage layer: CRUD, tags, full-text search and its safety."""

import pytest

from kb_mcp.store import KnowledgeStore, NoteNotFound, normalize_tags


@pytest.fixture
def store():
    s = KnowledgeStore()
    yield s
    s.close()


def test_add_and_get_roundtrip(store):
    n = store.add("DuckDB tips", "union_by_name opens every file", ["DuckDB", "data eng", "duckdb"])
    got = store.get(n.id)
    assert got.title == "DuckDB tips"
    assert got.tags == ["data-eng", "duckdb"]  # normalised, de-duplicated, sorted
    assert got.url is None


def test_empty_title_rejected(store):
    with pytest.raises(ValueError):
        store.add("   ")


def test_invalid_tag_rejected():
    with pytest.raises(ValueError):
        normalize_tags(["no!good"])


def test_update_changes_only_given_fields(store):
    n = store.add("Title", "body", ["a"])
    u = store.update(n.id, body="new body")
    assert (u.title, u.body, u.tags) == ("Title", "new body", ["a"])
    u = store.update(n.id, tags=["b", "c"])
    assert u.tags == ["b", "c"]


def test_missing_note_raises(store):
    with pytest.raises(NoteNotFound):
        store.get(42)
    with pytest.raises(NoteNotFound):
        store.delete(42)


def test_delete_cascades_tags_and_search_index(store):
    n = store.add("Airflow", "TaskFlow API", ["airflow"])
    store.delete(n.id)
    assert store.tags() == []
    assert store.search("taskflow") == []


def test_search_ranks_title_matches_first_and_supports_prefixes(store):
    store.add("Cooking", "a note that mentions kafka once in passing")
    store.add("Kafka consumer groups", "partitions and offsets")
    hits = store.search("kaf")  # prefix match on the last word
    assert [h.note.title for h in hits] == ["Kafka consumer groups", "Cooking"]
    assert "[" in hits[0].snippet or hits[0].snippet


def test_search_reflects_updates(store):
    n = store.add("Old title", "nothing here")
    store.update(n.id, body="streaming with redpanda")
    assert [h.note.id for h in store.search("redpanda")] == [n.id]


@pytest.mark.parametrize("query", ['"', "AND OR NOT", "title:*", "(((", "*", "NEAR(a b)"])
def test_search_never_breaks_on_fts_syntax(store, query):
    store.add("Something", "text")
    store.search(query)  # must not raise


def test_tag_filter_and_pagination(store):
    for i in range(5):
        store.add(f"note {i}", tags=["x"] if i % 2 == 0 else ["y"])
    page, total = store.list(tag="x", limit=2)
    assert total == 3 and len(page) == 2
    page2, _ = store.list(tag="x", limit=2, offset=2)
    assert {n.id for n in page}.isdisjoint({n.id for n in page2})


def test_stats_split_notes_and_bookmarks(store):
    store.add("a note")
    store.add("a link", url="https://example.com", tags=["web"])
    assert store.stats() == {"notes": 1, "bookmarks": 1, "tags": 1, "embedded_chunks": 0}


def test_persists_to_disk(tmp_path):
    path = tmp_path / "kb.db"
    s = KnowledgeStore(path)
    s.add("persist me")
    s.close()
    assert KnowledgeStore(path).search("persist")[0].note.title == "persist me"

"""Phase 5: agent memory. Dedup, supersede, scoring, forgetting and review."""

from datetime import UTC, datetime, timedelta

import pytest
from mcp import Client

from kb_mcp import memory as M
from kb_mcp.embeddings import HashingEmbedder
from kb_mcp.server import create_server
from kb_mcp.store import KnowledgeStore


class Clock:
    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, days):
        self.now += timedelta(days=days)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def mem(clock):
    store = KnowledgeStore(embedder=HashingEmbedder())
    store.memory._clock = clock
    yield store.memory
    store.close()


def test_scoring_components():
    assert M.recency_weight(0) == 1.0
    assert M.recency_weight(M.HALF_LIFE_DAYS) == pytest.approx(0.5)
    assert M.score(1.0, 5, 0) == pytest.approx(1.0)
    assert M.score(0.0, 1, 10_000) == pytest.approx(M.W_IMPORTANCE / 5, abs=1e-6)


def test_validation(mem):
    for bad in [dict(content="  "), dict(content="ok fine", kind="gossip"), dict(content="ok fine", importance=9)]:
        with pytest.raises(ValueError):
            mem.remember(**bad)


def test_only_identical_text_merges_automatically(mem):
    o1, a, _ = mem.remember("Prefers concise answers", "preference", 3)
    o2, b, _ = mem.remember("prefers   CONCISE answers", "preference", 5)
    assert (o1, o2) == ("created", "merged") and a.id == b.id
    assert mem.get(a.id).importance == 5  # merge keeps the higher importance
    assert len(mem.list()) == 1


def test_similar_memory_is_not_saved_but_returned_for_review(mem):
    _, va, _ = mem.remember("Lives in Virginia near the coast", "fact")
    outcome, saved, similar = mem.remember("Lives in Maryland near the coast", "fact")
    assert outcome == "needs_review" and saved is None
    assert [m.id for m in similar] == [va.id]
    assert len(mem.list()) == 1  # nothing written yet
    # The assistant decides it is a changed fact:
    outcome, new, _ = mem.remember("Lives in Maryland near the coast", "fact", supersedes=va.id)
    assert outcome == "superseded" and [m.id for m in mem.list()] == [new.id]


def test_force_saves_a_genuinely_different_memory(mem):
    mem.remember("Uses Kafka for streaming at work", "fact")
    assert mem.remember("Uses Kafka for streaming at home", "fact")[0] == "needs_review"
    outcome, m, _ = mem.remember("Uses Kafka for streaming at home", "fact", force=True)
    assert outcome == "created" and len(mem.list()) == 2


def test_unrelated_memory_saves_directly(mem):
    mem.remember("Prefers dark mode", "preference")
    assert mem.remember("Has a dog named Max", "fact")[0] == "created"


def test_identical_text_of_different_kind_is_reviewed_not_merged(mem):
    mem.remember("Uses DuckDB", "fact")
    assert mem.remember("Uses DuckDB", "project")[0] == "needs_review"


def test_supersede_keeps_history_but_hides_old_version(mem):
    _, old, _ = mem.remember("Lives in Maryland", "fact", 3)
    outcome, new, _ = mem.remember("Lives in Virginia", "fact", 3, supersedes=old.id)
    assert outcome == "superseded"
    assert [m.id for m in mem.list()] == [new.id]
    assert [m.content for m in mem.history(new.id)] == ["Lives in Virginia", "Lives in Maryland"]
    assert all(r.memory.id != old.id for r in mem.recall("where does the user live"))
    with pytest.raises(ValueError):
        mem.remember("Lives in Texas", supersedes=old.id)  # can't replace an already-replaced memory
    with pytest.raises(M.MemoryNotFound):
        mem.remember("x y z", supersedes=999)


def test_recall_prefers_relevant_then_important_then_recent(mem, clock):
    mem.remember("Building a knowledge base MCP server in Python", "project", 4)
    mem.remember("Likes hiking on weekends", "fact", 2)
    hits = mem.recall("python mcp server")
    assert hits[0].memory.content.startswith("Building a knowledge base")
    # No query: importance and recency decide.
    _, old_important, _ = mem.remember("Targets data engineering roles", "project", 5)
    clock.advance(200)
    _, fresh_minor, _ = mem.remember("Watched a documentary", "event", 1)
    overall = [h.memory.id for h in mem.recall()]
    assert overall.index(old_important.id) < overall.index(fresh_minor.id)


def test_recency_breaks_ties(mem, clock):
    _, a, _ = mem.remember("Working on the Olist project", "project", 3)
    clock.advance(90)
    _, b, _ = mem.remember("Working on the NBA project", "project", 3, force=True)
    hits = mem.recall("working on project")
    assert hits[0].memory.id == b.id and hits[0].recency > hits[1].recency


def test_irrelevant_memories_are_filtered_when_querying(mem):
    mem.remember("Prefers dark mode", "preference")
    mem.remember("Uses Kafka for streaming", "fact")
    hits = mem.recall("kafka streaming")
    assert [h.memory.content for h in hits] == ["Uses Kafka for streaming"]


def test_forget_removes_whole_chain(mem):
    _, a, _ = mem.remember("Version one", "fact")
    _, b, _ = mem.remember("Version two", "fact", supersedes=a.id)
    mem.forget(b.id)
    assert mem.list(include_superseded=True) == []


def test_review_flags_duplicates_and_stale(mem, clock):
    mem.remember("Uses Kafka streaming partitions and brokers daily", "fact", 3)
    mem.remember("Uses Kafka streaming partitions and brokers often", "project", 3, force=True)
    mem.remember("Tried a new coffee place", "event", 1)
    clock.advance(400)
    r = mem.review()
    assert len(r["possible_duplicates"]) == 1
    assert [m.content for m in r["stale"]] == ["Tried a new coffee place"]


def test_works_without_an_embedding_model(clock):
    store = KnowledgeStore()
    store.memory._clock = clock
    store.memory.remember("Prefers tea over coffee", "preference")
    assert store.memory.recall("tea")[0].memory.content == "Prefers tea over coffee"
    assert store.memory.remember("prefers tea over coffee", "preference")[0] == "merged"


def test_memories_persist(tmp_path):
    path = tmp_path / "kb.db"
    s = KnowledgeStore(path, embedder=HashingEmbedder())
    s.memory.remember("Remember me", "fact")
    s.close()
    assert KnowledgeStore(path).memory.list()[0].content == "Remember me"


def test_export_includes_history(mem):
    _, a, _ = mem.remember("Old fact", "fact")
    mem.remember("New fact", "fact", supersedes=a.id)
    assert '"superseded_by"' in mem.export() and "Old fact" in mem.export()


@pytest.mark.anyio
async def test_memory_tools_end_to_end():
    async with Client(create_server(KnowledgeStore(embedder=HashingEmbedder()))) as c:
        r = await c.call_tool("remember", {"content": "Prefers concise answers", "kind": "preference",
                                           "importance": 4})
        created = r.structured_content
        assert created["outcome"] == "created"
        again = (await c.call_tool("remember", {"content": "Prefers concise answers", "kind": "preference"}))
        assert again.structured_content["outcome"] == "merged"
        review = (await c.call_tool("remember", {"content": "Prefers concise answers always", "kind": "preference"}))
        assert review.structured_content["outcome"] == "needs_review"
        assert review.structured_content["memory"] is None
        assert review.structured_content["similar"][0]["id"] == created["memory"]["id"]
        rec = (await c.call_tool("recall", {"query": "concise answers"})).structured_content
        assert rec["memories"][0]["id"] == created["memory"]["id"] and rec["memories"][0]["score"] > 0
        md = (await c.read_resource("kb://memories")).contents[0].text
        assert "## Preferences" in md and "Prefers concise answers" in md
        bad = await c.call_tool("remember", {"content": "abc def", "kind": "gossip"})
        assert bad.is_error
        missing = await c.call_tool("forget", {"memory_id": 999})
        assert missing.is_error and "No memory with id 999" in missing.content[0].text
        gone = (await c.call_tool("forget", {"memory_id": created["memory"]["id"]})).structured_content
        assert gone["deleted"] is True
        assert (await c.call_tool("list_memories", {})).structured_content["result"] == []
        prompt = (await c.get_prompt("start_with_memory", {"topic": "job search"})).messages[0].content.text
        assert "recall" in prompt and "job search" in prompt


def test_memory_threshold_check_runs():
    from kb_mcp import evaluate

    rows = evaluate.memory_thresholds(HashingEmbedder())
    assert len(rows) == len(evaluate.MEMORY_PAIRS) and all(-1.01 <= r["similarity"] <= 1.01 for r in rows)
    assert {r["relation"] for r in rows} == {"same", "conflict", "unrelated"}

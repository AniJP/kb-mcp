"""Phase 2: chunking, embedding storage, staleness, semantic ranking and the eval harness.

Uses HashingEmbedder (shared words -> similar vectors) so the tests are fast and offline.
The real model's quality is measured separately by `kb-mcp eval`.
"""

import numpy as np
import pytest

from kb_mcp import evaluate
from kb_mcp.embeddings import HashingEmbedder, OverlapReranker, chunk_text
from kb_mcp.store import KnowledgeStore, NoteNotFound


@pytest.fixture
def store():
    s = KnowledgeStore(embedder=HashingEmbedder())
    yield s
    s.close()


def test_short_note_is_one_chunk_long_note_overlaps():
    assert chunk_text("T", "a b c") == ["T\na b c"]
    body = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text("T", body, max_words=180, overlap=40)
    assert len(chunks) == 3 and all(c.startswith("T\n") for c in chunks)
    assert "w140" in chunks[0] and "w140" in chunks[1]  # overlap region appears in both
    assert chunks[-1].endswith("w399")


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder()
    m = e.embed_documents(["kafka partitions", "kafka partitions", ""])
    assert np.allclose(np.linalg.norm(m[:2], axis=1), 1.0)
    assert np.allclose(m[0], m[1])


def test_notes_are_embedded_on_write(store):
    n = store.add("Kafka", "consumer groups and partitions")
    assert store.stats()["embedded_chunks"] == 1
    assert store.stale_note_ids() == []
    store.update(n.id, body="now about brokers")
    assert store.stale_note_ids() == []  # re-embedded on update
    store.delete(n.id)
    assert store.stats()["embedded_chunks"] == 0


def test_semantic_ranking_and_tag_filter(store):
    k = store.add("Kafka consumer groups", "partitions offsets", ["streaming"])
    store.add("Pasta", "garlic olive oil", ["cooking"])
    hits = store.semantic_search("kafka partitions")
    assert hits[0].note.id == k.id and hits[0].score > hits[1].score
    assert [h.note.id for h in store.semantic_search("kafka", tag="cooking")] != [k.id]


def test_best_chunk_wins_for_long_notes(store):
    filler = " ".join(["lorem"] * 400)
    long = store.add("Long note", f"{filler} redpanda streaming broker {filler}")
    store.add("Other", "nothing relevant here")
    hit = store.semantic_search("redpanda broker")[0]
    assert hit.note.id == long.id and "redpanda" in hit.snippet


def test_model_change_marks_everything_stale(tmp_path):
    path = tmp_path / "kb.db"
    s1 = KnowledgeStore(path, embedder=HashingEmbedder(name="model-a"))
    s1.add("one")
    s1.add("two")
    s1.close()
    s2 = KnowledgeStore(path, embedder=HashingEmbedder(name="model-b"))
    assert len(s2.stale_note_ids()) == 2
    assert s2.reindex() == 2 and s2.stale_note_ids() == []


def test_existing_notes_get_embedded_lazily(tmp_path):
    path = tmp_path / "kb.db"
    plain = KnowledgeStore(path)  # Phase 1 database, no embeddings
    plain.add("legacy note", "saved before semantic search existed")
    plain.close()
    s = KnowledgeStore(path, embedder=HashingEmbedder())
    assert s.semantic_search("legacy")[0].note.title == "legacy note"


class FailingEmbedder(HashingEmbedder):
    def embed_documents(self, texts):
        raise RuntimeError("no model offline")


def test_embedding_failure_never_loses_a_write():
    s = KnowledgeStore(embedder=FailingEmbedder())
    n = s.add("saved anyway")
    assert s.get(n.id).title == "saved anyway"
    assert s.stale_note_ids() == [n.id]


def test_similar_excludes_self_and_checks_id(store):
    a = store.add("Kafka groups", "partitions offsets consumers")
    b = store.add("Kafka partitions", "partitions offsets brokers")
    store.add("Pasta", "garlic oil")
    sim = store.similar(a.id)
    assert sim[0].note.id == b.id and a.id not in [h.note.id for h in sim]
    with pytest.raises(NoteNotFound):
        store.similar(999)


def test_eval_harness_runs_and_scores():
    data = evaluate.load_dataset()
    results = evaluate.run(embedder=HashingEmbedder(), reranker=OverlapReranker(), data=data)
    assert set(results) == {"keyword", "semantic", "hybrid", "hybrid kw=0.5", "hybrid+rerank"}
    for mode in results:
        s = evaluate.summarise(results[mode])
        assert s["all"]["queries"] == len(data["queries"])
        assert 0.0 <= s["all"]["mrr"] <= 1.0
    # Exact-term queries are what keyword search is built for.
    assert evaluate.summarise(results["keyword"])["exact"]["recall@5"] >= 0.8


def test_eval_dataset_is_consistent():
    data = evaluate.load_dataset()
    keys = [n["key"] for n in data["notes"]]
    assert len(keys) == len(set(keys))
    assert all(r in keys for q in data["queries"] for r in q["relevant"])
    assert {q["type"] for q in data["queries"]} == {"paraphrase", "exact", "identifier", "fine-grained"}


def test_ndcg():
    assert evaluate.ndcg_at_k(["a", "b"], {"a"}, 5) == 1.0
    assert evaluate.ndcg_at_k(["b", "a"], {"a"}, 5) == pytest.approx(1 / np.log2(3))
    assert evaluate.ndcg_at_k(["x"], {"a"}, 5) == 0.0


def test_any_terms_keyword_search_is_broader(store):
    store.add("Kafka retention", "segments deleted after retention.ms")
    assert store.search("kafka lost messages") == []  # every word required
    assert store.search("kafka lost messages", any_terms=True)[0].note.title == "Kafka retention"


def test_hybrid_rescues_identifier_queries(store):
    """An identifier that embeddings treat as noise is still found through the keyword list."""
    target = store.add("Postgres error 23505", "unique_violation on insert")
    for i in range(8):
        store.add(f"Postgres note {i}", "postgres error insert tips")
    assert store.hybrid_search("23505")[0].note.id == target.id


def test_rrf_rewards_agreement(store):
    both = store.add("Kafka consumer groups", "kafka partitions consumer groups")
    store.add("Kafka only in title", "nothing else")
    store.add("Groups", "consumer groups for a book club")
    hits = store.hybrid_search("kafka consumer groups")
    assert hits[0].note.id == both.id
    assert hits[0].score > hits[1].score


def test_reranker_reorders_top_candidates(store):
    store.add("Alpha", "kafka")
    best = store.add("Beta", "kafka retention segments deleted")
    hits = store.hybrid_search("kafka retention segments deleted", reranker=OverlapReranker())
    assert hits[0].note.id == best.id


def test_any_terms_ignores_function_words(store):
    store.add("My notes on cooking", "my my my")
    target = store.add("DuckDB memory", "ran out of RAM reading files")
    hits = store.search("my query ran out of RAM", any_terms=True)
    assert hits[0].note.id == target.id


def test_all_stopword_query_still_searches(store):
    store.add("How to", "how to do it")
    assert store.search("how to", any_terms=True)


def test_keyword_weight_scales_the_keyword_list(store):
    store.add("Alpha", "zebra")
    store.add("Beta", "zebra zebra")
    full = store.hybrid_search("zebra")
    half = store.hybrid_search("zebra", keyword_weight=0.5)
    assert half[0].score < full[0].score

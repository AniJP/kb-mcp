"""Retrieval evaluation: which search mode finds the right notes most often?

Loads a hand-labelled set (src/kb_mcp/eval_data.json) into a throwaway store and measures,
for each search mode:

  recall@5  share of a query's relevant notes that appear in the top 5 (averaged over queries)
  MRR       mean reciprocal rank of the first relevant result (1.0 = always first)
  nDCG@5    rank-weighted quality of the top 5 (1.0 = all relevant notes, at the top)
  latency   median milliseconds per query on this machine

Results are split by query type: paraphrase, exact, identifier and fine-grained (see the
dataset description). Run with `kb-mcp eval`.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from importlib import resources
from pathlib import Path

from kb_mcp.embeddings import Embedder, FastEmbedEmbedder, FastEmbedReranker, Reranker
from kb_mcp.store import KnowledgeStore, SearchHit


def load_dataset(path: str | Path | None = None) -> dict:
    if path is not None:
        return json.loads(Path(path).read_text())
    return json.loads(resources.files("kb_mcp").joinpath("eval_data.json").read_text())


def build_store(data: dict, embedder: Embedder | None) -> tuple[KnowledgeStore, dict[int, str]]:
    store = KnowledgeStore(embedder=embedder)
    id_to_key = {}
    for n in data["notes"]:
        note = store.add(n["title"], n["body"], n.get("tags"))
        id_to_key[note.id] = n["key"]
    return store, id_to_key


GROUPS = ["all", "paraphrase", "exact", "identifier", "fine-grained"]


def ndcg_at_k(keys: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, key in enumerate(keys[:k]) if key in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def score_queries(data: dict, search, id_to_key: dict[int, str], k: int = 5) -> list[dict]:
    rows = []
    for q in data["queries"]:
        start = time.perf_counter()
        hits: list[SearchHit] = search(q["q"])
        ms = (time.perf_counter() - start) * 1000
        keys = [id_to_key[h.note.id] for h in hits]
        relevant = set(q["relevant"])
        found = relevant & set(keys[:k])
        first = next((i for i, key in enumerate(keys) if key in relevant), None)
        rows.append({
            "query": q["q"],
            "type": q["type"],
            "recall_at_k": len(found) / len(relevant),
            "reciprocal_rank": 0.0 if first is None else 1.0 / (first + 1),
            "ndcg_at_k": ndcg_at_k(keys, relevant, k),
            "ms": ms,
            "top": keys[:3],
        })
    return rows


def summarise(rows: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for group in GROUPS:
        sel = [r for r in rows if group == "all" or r["type"] == group]
        if sel:
            out[group] = {
                "queries": len(sel),
                "recall@5": sum(r["recall_at_k"] for r in sel) / len(sel),
                "mrr": sum(r["reciprocal_rank"] for r in sel) / len(sel),
                "ndcg@5": sum(r["ndcg_at_k"] for r in sel) / len(sel),
                "median_ms": statistics.median(r["ms"] for r in sel),
            }
    return out


def run(
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    k: int = 5,
    data: dict | None = None,
    include_rerank: bool = True,
) -> dict[str, list[dict]]:
    data = data or load_dataset()
    embedder = embedder or FastEmbedEmbedder()
    store, id_to_key = build_store(data, embedder)
    store.semantic_search("warm up")  # load the model before timing anything
    modes = {
        "keyword": lambda q: store.search(q, limit=10),
        "semantic": lambda q: store.semantic_search(q, limit=10),
        "hybrid": lambda q: store.hybrid_search(q, limit=10),
        "hybrid kw=0.5": lambda q: store.hybrid_search(q, limit=10, keyword_weight=0.5),
    }
    if include_rerank:
        reranker = reranker or FastEmbedReranker()
        reranker.score("warm up", ["warm up"])
        modes["hybrid+rerank"] = lambda q: store.hybrid_search(q, limit=10, reranker=reranker)
    results = {name: score_queries(data, fn, id_to_key, k) for name, fn in modes.items()}
    store.close()
    return results


def print_report(results: dict[str, list[dict]]) -> None:
    summaries = {mode: summarise(rows) for mode, rows in results.items()}
    for metric in ["recall@5", "mrr", "ndcg@5"]:
        print(f"\n{metric}")
        print(f"{'query type':<14}{'n':>4}" + "".join(f"{m:>15}" for m in summaries))
        for g in GROUPS:
            n = summaries["keyword"][g]["queries"]
            print(f"{g:<14}{n:>4}" + "".join(f"{s[g][metric]:>15.2f}" for s in summaries.values()))
    print("\nmedian latency per query (ms)")
    print(f"{'':<18}" + "".join(f"{m:>15}" for m in summaries))
    print(f"{'all':<18}" + "".join(f"{s['all']['median_ms']:>15.1f}" for s in summaries.values()))
    best = max(summaries, key=lambda m: summaries[m]["all"]["mrr"])
    misses = [r for r in results[best] if r["reciprocal_rank"] == 0 or r["reciprocal_rank"] < 0.5]
    if misses:
        print(f"\n{best}: queries where the right note was not ranked first or second")
        for r in misses:
            print(f"  - [{r['type']}] {r['query']!r} -> top 3: {r['top']}")


def main() -> None:
    print_report(run())


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- memory thresholds (Phase 5)

MEMORY_PAIRS = [
    # (a, b, relation): same = one fact reworded; conflict = the fact changed; unrelated
    ("Prefers concise answers", "Likes short, concise replies", "same"),
    ("Applying for data engineering jobs", "Is job hunting for data engineer roles", "same"),
    ("Uses Python for most projects", "Writes most of their code in Python", "same"),
    ("Lives in Virginia", "Lives in Maryland", "conflict"),
    ("Prefers concise answers", "Prefers detailed explanations", "conflict"),
    ("Uses DuckDB for analytics", "Uses Postgres for analytics", "conflict"),
    ("Working on the NBA data platform", "Working on the Olist delivery project", "unrelated"),
    ("Has a dog named Max", "Likes hiking on weekends", "unrelated"),
]


def memory_thresholds(embedder: Embedder | None = None) -> list[dict]:
    """Cosine similarity of hand-picked memory pairs against the review threshold.

    'same' and 'conflict' pairs must both be surfaced for review (only the assistant can tell them
    apart); 'unrelated' pairs must not be.
    """
    from kb_mcp.memory import SIMILAR_THRESHOLD

    embedder = embedder or FastEmbedEmbedder()
    rows = []
    for a, b, relation in MEMORY_PAIRS:
        va, vb = embedder.embed_documents([a, b])
        sim = float(va @ vb)
        rows.append({"a": a, "b": b, "similarity": sim, "relation": relation,
                     "surfaced": sim >= SIMILAR_THRESHOLD, "should_surface": relation != "unrelated"})
    return rows


def print_memory_thresholds() -> None:
    from kb_mcp.memory import SIMILAR_THRESHOLD

    rows = memory_thresholds()
    print(f"review threshold {SIMILAR_THRESHOLD} (only identical text is merged automatically)\n")
    print(f"{'sim':>6}  {'relation':<10}{'surfaced':<10}pair")
    for r in rows:
        flag = "" if r["surfaced"] == r["should_surface"] else "  <-- wrong"
        print(f"{r['similarity']:>6.3f}  {r['relation']:<10}{str(r['surfaced']):<10}{r['a']!r} / {r['b']!r}{flag}")
    same = [r["similarity"] for r in rows if r["relation"] == "same"]
    conflict = [r["similarity"] for r in rows if r["relation"] == "conflict"]
    overlap = " overlapping, so similarity alone cannot decide merges" if max(conflict) >= min(same) else ""
    print(f"\nsame-fact pairs {min(same):.2f}-{max(same):.2f}, "
          f"conflicting pairs {min(conflict):.2f}-{max(conflict):.2f}:{overlap}")

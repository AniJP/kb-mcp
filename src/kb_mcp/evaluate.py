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

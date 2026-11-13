"""Retrieval evaluation: does semantic search actually find the right notes more often?

Loads a small hand-labelled set (src/kb_mcp/eval_data.json) into a throwaway store and
measures, for each search mode:

  recall@k  share of queries whose relevant notes appear in the top k results (averaged)
  MRR       mean reciprocal rank of the first relevant result (1.0 = always first)

Results are reported overall and split by query type: 'paraphrase' (different words, same
meaning) and 'exact' (a specific name or term). Run with `kb-mcp eval`.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from kb_mcp.embeddings import Embedder, FastEmbedEmbedder
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


def score_queries(data: dict, search, id_to_key: dict[int, str], k: int = 5) -> list[dict]:
    rows = []
    for q in data["queries"]:
        hits: list[SearchHit] = search(q["q"])
        keys = [id_to_key[h.note.id] for h in hits]
        relevant = set(q["relevant"])
        found = relevant & set(keys[:k])
        first = next((i for i, key in enumerate(keys) if key in relevant), None)
        rows.append({
            "query": q["q"],
            "type": q["type"],
            "recall_at_k": len(found) / len(relevant),
            "reciprocal_rank": 0.0 if first is None else 1.0 / (first + 1),
            "top": keys[:3],
        })
    return rows


def summarise(rows: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for group in ["all", "paraphrase", "exact"]:
        sel = [r for r in rows if group == "all" or r["type"] == group]
        if sel:
            out[group] = {
                "queries": len(sel),
                "recall@5": sum(r["recall_at_k"] for r in sel) / len(sel),
                "mrr": sum(r["reciprocal_rank"] for r in sel) / len(sel),
            }
    return out


def run(embedder: Embedder | None = None, k: int = 5, data: dict | None = None) -> dict[str, dict]:
    data = data or load_dataset()
    embedder = embedder or FastEmbedEmbedder()
    store, id_to_key = build_store(data, embedder)
    results = {
        "keyword": score_queries(data, lambda q: store.search(q, limit=10), id_to_key, k),
        "semantic": score_queries(data, lambda q: store.semantic_search(q, limit=10), id_to_key, k),
    }
    store.close()
    return results


def main() -> None:
    results = run()
    print(f"{'mode':<10}{'queries':<14}{'n':>4}{'recall@5':>10}{'MRR':>8}")
    for mode, rows in results.items():
        for group, m in summarise(rows).items():
            print(f"{mode:<10}{group:<14}{m['queries']:>4}{m['recall@5']:>10.2f}{m['mrr']:>8.2f}")
    misses = [r for r in results["semantic"] if r["reciprocal_rank"] == 0]
    if misses:
        print("\nsemantic misses (top 3 returned):")
        for r in misses:
            print(f"  - {r['query']!r}: {r['top']}")


if __name__ == "__main__":
    main()

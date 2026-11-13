# kb-mcp — a personal knowledge-base MCP server

An [MCP](https://modelcontextprotocol.io) server that gives an AI assistant a long-term memory you control: save notes and bookmarks, find them with **hybrid search** (meaning + exact words, optionally re-ranked by a cross-encoder), and pull them into a conversation. It is built from scratch on the official MCP Python SDK (v2), with SQLite storage, a local embedding model (no API key, nothing leaves your machine), and tests at the protocol level.

> **Status:** Phases 1–3 (MCP fundamentals, semantic search, hybrid search + evaluation) complete. See the [roadmap](#roadmap).

## What the assistant gets

**Tools.** Every tool declares annotations so clients can auto-approve reads and confirm destructive calls. Every tool returns typed, schema-checked output.

| Tool | Does | Annotations |
|---|---|---|
| `search_notes(query, tag?, limit?, mode?)` | `hybrid` (default): keyword + semantic fused with Reciprocal Rank Fusion, optionally re-ranked. `semantic`: meaning only. `keyword`: exact words, BM25. | read-only |
| `find_similar(note_id, limit?)` | Notes closest in meaning to a given note | read-only |
| `get_note(note_id)` | One note in full | read-only |
| `list_notes(tag?, limit?, offset?)` | Recent notes, paginated with `next_offset` | read-only |
| `list_tags()` | Tags with counts | read-only |
| `add_note(title, body?, tags?)` | Save a note | write |
| `save_bookmark(url, title, notes?, tags?)` | Save a link (http/https only) | write |
| `update_note(note_id, title?, body?, tags?)` | Change only the fields given | write, idempotent |
| `delete_note(note_id)` | Permanent delete | **destructive** |

**Resources**, which the user or client can attach as context:
- `kb://notes/{note_id}`: a note rendered as Markdown
- `kb://tags`: every tag with its note count
- `kb://stats`: counts of notes, bookmarks and tags

**Prompts**, as reusable workflows:
- `summarize_tag(tag)`: summarise everything under a topic, with citations
- `find_related(note_id)`: have the model search for connected notes and explain the links

## How semantic search works

1. **Chunk.** Each note is split into overlapping ~180-word passages, each prefixed with the note's title. Short notes stay as one chunk. This keeps a long note's different topics from blurring into one vector.
2. **Embed.** Each chunk goes through [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), a 384-dimensional model of about 70 MB, run locally with ONNX Runtime via `fastembed`. Vectors are L2-normalised.
3. **Store.** Vectors live in SQLite next to the notes (`note_chunks`), tagged with the model name and a hash of the note's content.
4. **Search.** The query is embedded, scored against every chunk with one matrix multiplication (cosine similarity), and each note is ranked by its best chunk.

Engineering choices:
- **Brute-force search, on purpose.** A personal knowledge base has thousands of chunks, not millions. Exact search over a NumPy matrix takes about a millisecond and is never wrong, so an approximate index (FAISS, HNSW) would add complexity for no gain at this scale. The store has a single place to swap one in.
- **Always consistent.** Notes are re-embedded when saved or edited. A content hash and the model name detect anything stale, including notes saved before Phase 2 existed or after a model change, and fix it before searching.
- **Fails soft.** If the model can't load (for example, offline on first run), saving still works, and search falls back to keyword mode with a `notice` saying so, rather than erroring.

## How hybrid search works (Phase 3)

Keyword and semantic search fail in opposite places:
- **Embeddings** blur codes and identifiers. `23505`, `B905` and `HEAD@{2}` carry almost no meaning.
- **Keyword search** misses anything phrased differently.

Hybrid search runs both and merges them:
1. **Two candidate lists.** BM25 in *any-term* mode (any word may match, so it finds more candidates) and semantic search each return their top 30.
2. **Reciprocal Rank Fusion.** Each note scores `Σ 1/(60 + rank)` across the two lists. RRF uses ranks, not raw scores, so BM25 and cosine similarity never need calibrating against each other. A note found by both lists rises highest.
3. **Optional re-ranking** (`kb-mcp --rerank`). A cross-encoder ([ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), about 80 MB, local) reads the query and each of the top 20 candidates *together* and re-orders them. It is more accurate than comparing vectors and much slower, so it only touches the shortlist.

Every step fails soft:
- No embedding model: keyword results, with a notice.
- No re-ranker: fused results, with a notice.

## Measuring it

`kb-mcp eval` loads a hand-labelled set ([`eval_data.json`](src/kb_mcp/eval_data.json): 52 notes, 60 queries) into a throwaway store and scores every mode on recall@5, MRR, nDCG@5 and latency. The queries are deliberately mixed so no single method can look perfect:

| Query type | n | Example | Hard for |
|---|---|---|---|
| paraphrase | 31 | "my container keeps getting killed" → the exit-code-137 note | keyword |
| exact | 7 | "XCom" | — |
| identifier | 11 | "23505", "B905", "0 6 * * 1-5" | embeddings |
| fine-grained | 11 | "why did old messages disappear from my topic" among four near-identical Kafka notes | both |

**Results** (MacBook Air, Apple Silicon; MRR = how often the right note comes first, 1.0 = always):

| Query type | n | keyword | semantic | **hybrid** (default) | hybrid + re-rank |
|---|---|---|---|---|---|
| paraphrase | 31 | 0.00 | **0.91** | 0.85 | 0.87 |
| exact | 7 | 1.00 | 1.00 | 1.00 | 1.00 |
| identifier | 11 | 1.00 | 1.00 | 1.00 | 1.00 |
| fine-grained | 11 | 0.00 | 0.79 | 0.85 | **0.88** |
| **all (MRR)** | 60 | 0.30 | **0.92** | 0.90 | 0.91 |
| all (recall@5) | 60 | 0.30 | 0.97 | 0.97 | 0.97 |
| all (nDCG@5) | 60 | 0.30 | 0.92 | 0.91 | 0.92 |
| median latency | | 0.1 ms | 2.5 ms | 2.7 ms | 79 ms |

**What the evaluation changed**
- **The first version of hybrid search was worse than semantic alone** (MRR 0.88 vs 0.92, and 0.82 vs 0.91 on paraphrases). Its misses pointed to the cause: in a small corpus, function words like "my" and "how" are *rare*, so BM25 gave them high weight, and "my query ran out of RAM" matched notes that happened to contain "my". Dropping stopwords from the keyword half raised hybrid to recall@5 0.97 and MRR 0.90. A test now guards against the regression.
- **Down-weighting the keyword list (0.5×) changed nothing**, so the problem was noisy keyword candidates, not the fusion weights.

**Conclusions**
- **Semantic and hybrid are statistically tied.** The 0.02 MRR gap is about one query out of 60. Each wins where you'd expect: semantic on paraphrases, hybrid on near-duplicates.
- **Hybrid stays the default.** It equals semantic on recall, is better at telling near-identical notes apart, and keeps an exact-match path for codes in a larger, messier notebook. In this set the embeddings happened to handle every identifier anyway.
- **Re-ranking is optional** (`--rerank`). It is the best mode on near-duplicates (recall@5 1.00, MRR 0.88) but about 30× slower and no better overall.
- **Caveat:** 60 hand-written queries is small, and the dataset was extended while this phase was being built. Treat these as directional, not definitive.

## Design decisions

- **The storage layer knows nothing about MCP.** `store.py` is plain Python + SQLite, and `server.py` is a thin protocol layer over it. Later phases (vector search, ingestion) plug in behind the same interface.
- **Errors are part of the contract.** An unknown id, a bad URL or an invalid tag raises `ToolError`, so the model gets an actionable message ("No note with id 999. Use search_notes…"), not a stack trace. Missing resources return a proper `ResourceNotFound` protocol error.
- **Search is injection-proof.** User text is tokenised and quoted before it reaches FTS5, so stray quotes or `AND/OR/NEAR` can never break or alter the query. This is tested.
- **The full-text index stays consistent by construction.** It is an external-content FTS5 table kept in sync by triggers, so updates and deletes can't leave stale search results.
- **Two transports:**
  - stdio, for Claude Desktop and other local clients
  - Streamable HTTP (`--http`), for remote or multi-client use

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                         # 52 tests: storage, semantic + hybrid search, protocol, real stdio round-trip
kb-mcp reindex                 # one-time: downloads the embedding model and embeds existing notes
kb-mcp eval                    # keyword vs semantic vs hybrid vs re-ranked, by query type
kb-mcp --rerank                # serve with cross-encoder re-ranking on
kb-mcp --db ~/.kb-mcp/kb.db    # stdio
kb-mcp --http --port 8765      # Streamable HTTP at http://127.0.0.1:8765/mcp
```

### Use it from Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), using the absolute path to the `kb-mcp` script in your virtualenv:

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/ABSOLUTE/PATH/TO/.venv/bin/kb-mcp",
      "args": ["--db", "/Users/YOU/.kb-mcp/kb.db"]
    }
  }
}
```

Restart Claude Desktop, then try *"save a note that DuckDB's union_by_name opens every file at once"* and *"what have I saved about DuckDB?"*

### Inspect it

```bash
npx @modelcontextprotocol/inspector .venv/bin/kb-mcp
```

## Tests

| File | What it proves |
|---|---|
| `tests/test_store.py` | CRUD, tag normalisation, BM25 ranking (title > body), prefix search, index consistency after update and delete, FTS-syntax safety, persistence |
| `tests/test_semantic.py` | Chunking, embedding on write, staleness after edits and model changes, legacy-database backfill, best-chunk ranking, tag filter, similarity, failure never losing a write, RRF fusion, identifier rescue, re-ranking, eval harness and dataset integrity |
| `tests/test_server.py` | Through a real MCP client (in-process): tool schemas and annotations, structured output, error contract, input validation, pagination, resources, prompts, search modes, fallback notices |
| `tests/test_stdio.py` | The server launched as a subprocess over stdio, the way Claude Desktop runs it |

## Roadmap

| Phase | Adds | Status |
|---|---|---|
| 1. MCP fundamentals | Tools, resources, prompts, annotations, structured output, stdio + HTTP | ✅ |
| 2. Semantic search | Local embeddings, chunking, vector search, `find_similar`, retrieval eval | ✅ |
| 3. Hybrid search | BM25 + vectors with RRF, cross-encoder re-ranking, harder evaluation (identifier and near-duplicate queries) | ✅ |
| 4. Ingestion pipeline | Import from files, web pages and bookmarks exports; chunking, dedup, scheduled refresh | |
| 5. Agent memory | Memories the assistant writes itself, with recency and importance scoring | |

## Layout

```
src/kb_mcp/store.py        SQLite storage: notes, FTS5 keyword index, embeddings (no MCP dependency)
src/kb_mcp/embeddings.py   embedding + re-ranking models (local fastembed, offline test doubles) and chunking
src/kb_mcp/evaluate.py     retrieval evaluation (recall@5, MRR, nDCG@5, latency) on eval_data.json
src/kb_mcp/server.py       MCP tools, resources, prompts, CLI entry point
tests/                 storage, protocol and stdio tests
```

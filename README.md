# kb-mcp — a personal knowledge-base MCP server

An [MCP](https://modelcontextprotocol.io) server that gives an AI assistant a long-term memory you control: save notes and bookmarks, find them by **meaning** or by exact words, and pull them into a conversation. It is built from scratch on the official MCP Python SDK (v2), with SQLite storage, a local embedding model (no API key, nothing leaves your machine), and tests at the protocol level.

> **Status:** Phases 1 (MCP fundamentals) and 2 (semantic search) complete. See the [roadmap](#roadmap).

## What the assistant gets

**Tools.** Every tool declares annotations so clients can auto-approve reads and confirm destructive calls. Every tool returns typed, schema-checked output.

| Tool | Does | Annotations |
|---|---|---|
| `search_notes(query, tag?, limit?, mode?)` | `semantic` (default): match by meaning with local embeddings. `keyword`: exact words, BM25-ranked. | read-only |
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

## Measuring it

`kb-mcp eval` loads a small hand-labelled set ([`eval_data.json`](src/kb_mcp/eval_data.json): 32 notes, 32 queries) into a throwaway store and compares the two modes. *Paraphrase* queries share almost no words with the note they should find ("my query ran out of RAM" → the DuckDB `union_by_name` note); *exact* queries use a specific term ("XCom").

| Mode | Query type | Queries | Recall@5 | MRR |
|---|---|---|---|---|
| keyword | paraphrase | 25 | 0.00 | 0.00 |
| keyword | exact | 7 | 1.00 | 1.00 |
| **semantic** | **paraphrase** | 25 | **1.00** | **0.90** |
| semantic | exact | 7 | 1.00 | 1.00 |
| keyword | all | 32 | 0.22 | 0.22 |
| **semantic** | **all** | 32 | **1.00** | **0.92** |

**Reading it:**
- Keyword search finds nothing for paraphrases. That's partly by design: it requires every query word to appear, which keeps exact lookups precise.
- Semantic search puts the right note in the top 5 for every query, and first for most of them (MRR 0.92). It loses nothing on exact-term queries.
- **Caveat:** 32 hand-written queries is a small, friendly test. Phase 3 builds a larger and harder evaluation set, where keyword and semantic each fail on different queries, to justify combining them.

*Model: BAAI/bge-small-en-v1.5 via fastembed, run on a MacBook Air (Apple Silicon).*

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
pytest                         # 41 tests: storage, semantic search, protocol, real stdio round-trip
kb-mcp reindex                 # one-time: downloads the embedding model and embeds existing notes
kb-mcp eval                    # keyword vs semantic retrieval quality
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
| `tests/test_semantic.py` | Chunking, embedding on write, staleness after edits and model changes, legacy-database backfill, best-chunk ranking, tag filter, similarity, failure never losing a write, eval harness |
| `tests/test_server.py` | Through a real MCP client (in-process): tool schemas and annotations, structured output, error contract, input validation, pagination, resources, prompts, search modes, fallback notices |
| `tests/test_stdio.py` | The server launched as a subprocess over stdio, the way Claude Desktop runs it |

## Roadmap

| Phase | Adds | Status |
|---|---|---|
| 1. MCP fundamentals | Tools, resources, prompts, annotations, structured output, stdio + HTTP | ✅ |
| 2. Semantic search | Local embeddings, chunking, vector search, `find_similar`, retrieval eval | ✅ |
| 3. Hybrid search | BM25 + vectors with rank fusion and re-ranking; retrieval evaluation set | |
| 4. Ingestion pipeline | Import from files, web pages and bookmarks exports; chunking, dedup, scheduled refresh | |
| 5. Agent memory | Memories the assistant writes itself, with recency and importance scoring | |

## Layout

```
src/kb_mcp/store.py        SQLite storage: notes, FTS5 keyword index, embeddings (no MCP dependency)
src/kb_mcp/embeddings.py   embedding models (local fastembed + an offline test double) and chunking
src/kb_mcp/evaluate.py     retrieval evaluation (recall@5, MRR) on eval_data.json
src/kb_mcp/server.py       MCP tools, resources, prompts, CLI entry point
tests/                 storage, protocol and stdio tests
```

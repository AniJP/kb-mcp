# kb-mcp — a personal knowledge-base MCP server

An [MCP](https://modelcontextprotocol.io) server that gives an AI assistant a long-term memory you control: save notes and bookmarks, search them with full-text ranking, and pull them into a conversation. It is built from scratch on the official MCP Python SDK (v2), with SQLite + FTS5 storage, and tested at the protocol level.

> **Status:** Phase 1 (MCP fundamentals) complete. See the [roadmap](#roadmap) for RAG, hybrid search, ingestion and agent memory.

## What the assistant gets

**Tools.** Every tool declares annotations so clients can auto-approve reads and confirm destructive calls. Every tool returns typed, schema-checked output.

| Tool | Does | Annotations |
|---|---|---|
| `search_notes(query, tag?, limit?)` | BM25-ranked full-text search with highlighted snippets | read-only |
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
pytest                         # 27 tests: storage, protocol, real stdio round-trip
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
| `tests/test_server.py` | Through a real MCP client (in-process): tool schemas and annotations, structured output, error contract, input validation, pagination, resources, prompts |
| `tests/test_stdio.py` | The server launched as a subprocess over stdio, the way Claude Desktop runs it |

## Roadmap

| Phase | Adds | Status |
|---|---|---|
| 1. MCP fundamentals | Tools, resources, prompts, annotations, structured output, stdio + HTTP | ✅ |
| 2. Semantic search | Embeddings + vector index (RAG) alongside FTS | |
| 3. Hybrid search | BM25 + vectors with rank fusion and re-ranking; retrieval evaluation set | |
| 4. Ingestion pipeline | Import from files, web pages and bookmarks exports; chunking, dedup, scheduled refresh | |
| 5. Agent memory | Memories the assistant writes itself, with recency and importance scoring | |

## Layout

```
src/kb_mcp/store.py    SQLite + FTS5 storage (no MCP dependency)
src/kb_mcp/server.py   MCP tools, resources, prompts, CLI entry point
tests/                 storage, protocol and stdio tests
```

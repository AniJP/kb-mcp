# kb-mcp — a personal knowledge-base MCP server

An [MCP](https://modelcontextprotocol.io) server that gives an AI assistant a long-term memory you control: save notes and bookmarks, import folders and web pages, find anything with **hybrid search**, and let the assistant **remember what it learns about you** across conversations. It is built from scratch on the official MCP Python SDK (v2), with SQLite storage, a local embedding model (no API key, nothing leaves your machine), and tests at the protocol level.

> **Status:** all five phases complete: MCP fundamentals, semantic search, hybrid search + evaluation, ingestion pipeline, agent memory. See the [roadmap](#roadmap).

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
| `import_url(url, tags?)` | Fetch a web page, keep its readable text, save it as a note; re-importing refreshes it | write, idempotent, **open-world** |
| `import_history(limit?)` | Recent import runs with counts and errors | read-only |
| `remember(content, kind?, importance?, supersedes?, force?)` | Keep a fact about the user. Similar existing memories are returned for review instead of being duplicated or overwritten | write |
| `recall(query?, kinds?, limit?)` | Memories ranked by relevance × importance × recency | read-only |
| `list_memories(kinds?, include_superseded?)` | Everything remembered | read-only |
| `review_memories()` | Possible duplicates and stale minor memories (changes nothing) | read-only |
| `forget(memory_id)` | Delete a memory and its earlier versions | **destructive** |

**Resources**, which the user or client can attach as context:
- `kb://notes/{note_id}`: a note rendered as Markdown
- `kb://tags`: every tag with its note count
- `kb://stats`: counts of notes, bookmarks and tags
- `kb://imports`: the last 20 import runs
- `kb://memories`: everything remembered, grouped by kind

**Prompts**, as reusable workflows:
- `summarize_tag(tag)`: summarise everything under a topic, with citations
- `find_related(note_id)`: have the model search for connected notes and explain the links
- `start_with_memory(topic?)`: open a conversation by recalling relevant memories

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

## Ingestion pipeline (Phase 4)

Bring in what you already have, and keep it in sync:

```bash
kb-mcp ingest folder ~/Notes            # Markdown / .txt (an Obsidian vault works as-is)
kb-mcp ingest folder ~/Notes --prune    # ...and remove notes whose file was deleted
kb-mcp ingest bookmarks ~/Downloads/bookmarks.html   # browser export: Chrome, Safari, Firefox, Edge
kb-mcp ingest url https://example.com/article --tag reading
kb-mcp runs                             # what each run added, updated, skipped or failed
```

| Source | Title | Tags | Identity (for re-runs) |
|---|---|---|---|
| Folder | front matter `title:` → first `# heading` → file name | front matter `tags:` + inline `#hashtags` | folder + relative path |
| Bookmarks | link text | bookmark folder names (browser root folders skipped) | normalised URL |
| Web page | `<title>` | given with `--tag` | normalised final URL |

How the pipeline is built:
- **Idempotent.** Each imported note is tracked by `(source type, source id)` with a content hash. A re-run updates only what changed and skips the rest. Unchanged items cost nothing, not even re-embedding.
- **No duplicates.** URLs are normalised: lower-case host, no `#fragment`, no `utm_*`/`fbclid` tracking parameters, no trailing slash. A bookmark for a page you already saved by hand is *linked* to your note instead of copied, and your note is left untouched.
- **Safe deletes.** `--prune` is opt-in and only removes notes that came from *that* folder. It never touches hand-written notes or other sources.
- **Partial failure is normal.** A bad file or a page that times out is counted and logged, and the run continues. Every run is recorded in `ingest_runs` and exposed through `import_history` and `kb://imports`.
- **SSRF-safe fetching.** `import_url` can be called by the model, so a prompt-injected page could try to make the server fetch `localhost`, a router or a cloud metadata endpoint. The fetcher therefore:
  - allows only http(s)
  - resolves the hostname and refuses non-public addresses, re-checking after every redirect
  - caps pages at 2 MB with a 15-second timeout
  - accepts only HTML or plain text

  Folder and bookmark imports stay CLI-only, so the model can never read arbitrary files from your disk.
- **Readable text only.** Scripts, styles, navigation, headers, footers and sidebars are dropped. `<article>`/`<main>` is preferred when it holds real content.

**Keeping a folder in sync automatically** (macOS/Linux cron, every morning at 7:45):
```bash
crontab -e
45 7 * * * /ABSOLUTE/PATH/TO/.venv/bin/kb-mcp ingest folder ~/Notes --prune >> ~/.kb-mcp/ingest.log 2>&1
```

## Agent memory (Phase 5)

Notes are documents you choose to save. **Memories** are short facts the assistant picks up on its own while you talk ("prefers concise answers", "applying for data engineering roles", "decided on DuckDB over Postgres"), so the next conversation starts with that context.

**Ranking.** Recall combines three signals, following the recency × importance × relevance idea from *Generative Agents* (Park et al., 2023):

```
score = 0.6 × relevance  +  0.25 × importance/5  +  0.15 × 0.5^(age_days / 30)
```

- *relevance* is the cosine similarity to the current topic. With no topic, everything counts as relevant, so "what do you know about me?" returns the most important and recent memories.
- *importance* (1–5) is set by the assistant when it saves the memory.
- *recency* halves every 30 days since the memory was last confirmed.

**Memories are never silently rewritten, and embeddings never decide what counts as "the same".** The first design merged any new memory with cosine ≥ 0.92 to an existing one. Measuring that threshold on the real model (`kb-mcp eval --memory`) showed the idea itself was wrong, not just the number:

| Pair | Relation | Cosine |
|---|---|---|
| "Applying for data engineering jobs" / "Is job hunting for data engineer roles" | same fact | 0.880 |
| "Lives in Virginia" / "Lives in Maryland" | **conflict** | **0.827** |
| "Prefers concise answers" / "Likes short, concise replies" | same fact | 0.819 |
| "Uses DuckDB for analytics" / "Uses Postgres for analytics" | **conflict** | **0.815** |
| "Prefers concise answers" / "Prefers detailed explanations" | **conflict** | **0.798** |
| "Uses Python for most projects" / "Writes most of their code in Python" | same fact | 0.783 |
| "Working on the NBA data platform" / "Working on the Olist delivery project" | unrelated | 0.634 |
| "Has a dog named Max" / "Likes hiking on weekends" | unrelated | 0.435 |

Rewordings of one fact (0.78–0.88) and *contradictory* facts (0.80–0.83) overlap completely, so any merge threshold either duplicates memories or silently overwrites "Maryland" with "Virginia". The design was changed so that:
- **Only identical text** (ignoring case and spacing, same kind) is merged automatically.
- **Anything similar (cosine ≥ 0.75) is not saved.** `remember` returns `needs_review` with the similar memories, and the assistant, which understands the sentences, decides. If it's the *same fact*, it does nothing. If the *fact changed*, it calls `remember(..., supersedes=<id>)`, which keeps the old version as history but hides it from recall. If it's a *different fact*, it calls `remember(..., force=true)`.
- At 0.75, every same-fact and conflicting pair above is surfaced for review, and both unrelated pairs are saved directly.
- `review_memories` uses the same threshold to list possible duplicates and stale minor memories. It deletes nothing itself.

**Privacy and control:**
- Memories live in the same local SQLite file as the notes.
- The server instructions tell the assistant not to store secrets, credentials or sensitive personal details unless asked.
- `forget` deletes a memory together with its history.
- `kb-mcp memories` lists everything, and `kb-mcp memories --export memories.json` exports it.

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
pytest                         # 90 tests: storage, search, ingestion, memory, protocol, real stdio round-trip
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
| `tests/test_ingest.py` | Folder parsing (front matter, headings, hashtags, hidden folders), idempotent re-runs, change detection with re-embedding, scoped prune, per-item failures, bookmark parsing, URL normalisation and dedup, readable-text extraction, SSRF guard |
| `tests/test_memory.py` | Validation, identical-text merging, needs-review for similar memories, force and supersede paths, supersede history, relevance/importance/recency ranking with a fake clock, filtering irrelevant memories, forgetting whole chains, review, no-model fallback, persistence, export, tools end to end |
| `tests/test_server.py` | Through a real MCP client (in-process): tool schemas and annotations, structured output, error contract, input validation, pagination, resources, prompts, search modes, fallback notices |
| `tests/test_stdio.py` | The server launched as a subprocess over stdio, the way Claude Desktop runs it |

## Roadmap

| Phase | Adds | Status |
|---|---|---|
| 1. MCP fundamentals | Tools, resources, prompts, annotations, structured output, stdio + HTTP | ✅ |
| 2. Semantic search | Local embeddings, chunking, vector search, `find_similar`, retrieval eval | ✅ |
| 3. Hybrid search | BM25 + vectors with RRF, cross-encoder re-ranking, harder evaluation (identifier and near-duplicate queries) | ✅ |
| 4. Ingestion pipeline | Folders, bookmark exports, web pages; idempotent upserts, URL dedup, scoped prune, run log, SSRF-safe fetching | ✅ |
| 5. Agent memory | Memories the assistant writes itself: model-judged dedup and supersede, recency × importance × relevance recall, review, forget, export | ✅ |

## Layout

```
src/kb_mcp/store.py        SQLite storage: notes, FTS5 keyword index, embeddings (no MCP dependency)
src/kb_mcp/embeddings.py   embedding + re-ranking models (local fastembed, offline test doubles) and chunking
src/kb_mcp/memory.py       agent memory: remember / recall / supersede / review / forget
src/kb_mcp/ingest.py       ingestion: folder, bookmarks and web sources, the idempotent pipeline, SSRF guard
src/kb_mcp/evaluate.py     retrieval evaluation (recall@5, MRR, nDCG@5, latency) on eval_data.json
src/kb_mcp/server.py       MCP tools, resources, prompts, CLI entry point
tests/                 storage, protocol and stdio tests
```

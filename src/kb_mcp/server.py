"""MCP protocol layer: exposes the knowledge store as tools, resources and prompts.

Design rules followed throughout:
  * Tools that change data are separate from tools that read it, and every tool declares
    `ToolAnnotations` so clients can tell read-only, idempotent and destructive calls apart.
  * Tools return typed (Pydantic) results, so clients get both readable text and a JSON
    `structuredContent` that matches the tool's advertised `outputSchema`.
  * Expected failures (unknown id, bad input) raise `ToolError` / `ResourceNotFoundError`,
    which reach the model as a clear error instead of a crash.
  * Read-only context the client can attach (a note, the tag list, stats) is a resource;
    reusable workflows are prompts.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Literal
from urllib.parse import urlparse

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.server.mcpserver.prompts.base import UserMessage
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from kb_mcp.embeddings import FastEmbedEmbedder, FastEmbedReranker, Reranker
from kb_mcp.ingest import fetch_html, ingest_url
from kb_mcp.memory import KINDS, Memory, MemoryNotFound
from kb_mcp.store import KnowledgeStore, Note, NoteNotFound, SearchHit

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
A personal knowledge base of notes and bookmarks.
- Use search_notes before answering questions about things the user saved; cite note ids.
  The default (hybrid) mode combines meaning and exact words; use mode="keyword" for exact names,
  codes or quotes only.
- Use find_similar to explore notes related to one the user is looking at.
- Use add_note / save_bookmark only when the user asks to save something.
- import_url fetches a web page into the knowledge base; use it when the user asks to save an article.
- delete_note is permanent: confirm with the user first.

Memory (separate from notes):
- At the start of a conversation, call recall with the topic at hand to load relevant context.
- Call remember when the user states something durable and useful later: a preference, an
  ongoing project, a decision, a stable fact about their situation. Keep each memory to one
  short sentence in third person ("Prefers concise answers").
- If remember answers needs_review, compare the new fact with each similar memory:
  same fact -> do nothing; the fact changed -> call remember again with supersedes=<old id>;
  genuinely different -> call remember again with force=true.
- If you already know a fact changed, pass supersedes=<old id> directly.
- Do not remember passing remarks, secrets, credentials or sensitive personal details
  (health, finances, identity documents) unless the user explicitly asks you to.
- When the user asks what you know about them, use recall without a query. When they ask you
  to forget something, use forget.
"""

READ = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
UPDATE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
DELETE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)
# Fetches from the internet: open world. Re-importing the same URL refreshes one note: idempotent.
IMPORT = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True)


# --------------------------------------------------------------------------- result models


class NoteOut(BaseModel):
    id: int
    title: str
    body: str
    url: str | None = Field(None, description="Set for bookmarks")
    tags: list[str]
    created_at: str
    updated_at: str
    uri: str = Field(description="Resource URI for this note")

    @classmethod
    def of(cls, n: Note) -> NoteOut:
        return cls(**n.__dict__, uri=f"kb://notes/{n.id}")


class Hit(BaseModel):
    id: int
    title: str
    snippet: str = Field(description="Matching text, with matches in [brackets]")
    score: float = Field(description="Relevance, higher is better")
    tags: list[str]
    url: str | None = None


class SearchResult(BaseModel):
    query: str
    mode: Literal["hybrid", "semantic", "keyword"] = Field(description="The search mode actually used")
    reranked: bool = Field(False, description="Whether a cross-encoder re-ordered the top results")
    hits: list[Hit]
    notice: str | None = Field(None, description="Set when the requested mode was unavailable and a fallback was used")


class SimilarResult(BaseModel):
    note_id: int
    similar: list[Hit]


class NotePage(BaseModel):
    notes: list[NoteOut]
    total: int = Field(description="Number of notes matching the filter")
    next_offset: int | None = Field(None, description="Pass as offset to get the next page; null on the last page")


class TagCount(BaseModel):
    tag: str
    count: int


class ImportResult(BaseModel):
    outcome: Literal["added", "updated", "unchanged", "linked"] = Field(
        description="added: new note. updated: page changed since last import. unchanged: same content. "
                    "linked: the URL was already saved, nothing duplicated."
    )
    note: NoteOut


class ImportRun(BaseModel):
    id: int
    source_type: str
    source: str
    started_at: str
    finished_at: str | None
    added: int
    updated: int
    unchanged: int
    linked: int
    pruned: int
    failed: int
    errors: list[str]


MemoryKind = Literal["preference", "fact", "project", "decision", "event"]
assert set(KINDS) == {"preference", "fact", "project", "decision", "event"}


class MemoryOut(BaseModel):
    id: int
    content: str
    kind: MemoryKind
    importance: int = Field(description="1 (minor) to 5 (essential)")
    created_at: str
    updated_at: str
    superseded_by: int | None = Field(None, description="Set when a newer memory replaced this one")

    @classmethod
    def of(cls, m: Memory) -> MemoryOut:
        return cls(**m.__dict__)


class RememberResult(BaseModel):
    outcome: Literal["created", "merged", "superseded", "needs_review"] = Field(
        description="created: saved. merged: identical memory existed and was refreshed. superseded: saved, "
                    "replacing `supersedes`. needs_review: NOT saved, see `similar` and decide."
    )
    memory: MemoryOut | None = Field(None, description="The saved memory (absent for needs_review)")
    similar: list[MemoryOut] = Field(
        default_factory=list,
        description="For needs_review: existing memories that may say the same thing or contradict the new one. "
                    "Same fact: do nothing. Changed fact: call remember with supersedes=<id>. "
                    "Different fact: call remember with force=true.",
    )


class RecalledMemory(MemoryOut):
    score: float = Field(description="Combined relevance, importance and recency; higher is better")
    relevance: float
    recency: float


class RecallResult(BaseModel):
    query: str | None
    memories: list[RecalledMemory]


class DuplicatePair(BaseModel):
    a: MemoryOut
    b: MemoryOut
    similarity: float


class MemoryReview(BaseModel):
    possible_duplicates: list[DuplicatePair]
    stale: list[MemoryOut] = Field(description="Low-importance memories not updated in 6+ months")


class Deleted(BaseModel):
    id: int
    title: str
    deleted: bool = True


# --------------------------------------------------------------------------- server


Tags = Annotated[list[str], Field(description="Short labels such as 'python' or 'job-search'", max_length=20)]


def _hits(hits: list[SearchHit]) -> list[Hit]:
    return [
        Hit(id=h.note.id, title=h.note.title, snippet=h.snippet, score=h.score, tags=h.note.tags, url=h.note.url)
        for h in hits
    ]


def _not_found(e: NoteNotFound) -> ToolError:
    return ToolError(f"No note with id {e.note_id}. Use search_notes or list_notes to find the right id.")


def create_server(store: KnowledgeStore, reranker: Reranker | None = None, fetch=fetch_html) -> MCPServer:
    mcp = MCPServer(name="knowledge-base", version="0.5.0", instructions=INSTRUCTIONS)

    # ----------------------------------------------------------------- tools: read

    @mcp.tool(title="Search notes", annotations=READ)
    def search_notes(
        query: Annotated[str, Field(description="Words to look for in titles and bodies", min_length=1)],
        tag: Annotated[str | None, Field(description="Only search notes with this tag")] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        mode: Annotated[
            Literal["hybrid", "semantic", "keyword"],
            Field(description="hybrid (default): meaning + exact words combined. semantic: meaning only. "
                              "keyword: exact words only, ranked by BM25."),
        ] = "hybrid",
    ) -> SearchResult:
        """Search notes and bookmarks, best matches first."""
        notice = None
        try:
            if mode != "keyword" and store.embedder is not None:
                try:
                    if mode == "semantic":
                        return SearchResult(query=query, mode="semantic",
                                            hits=_hits(store.semantic_search(query, tag=tag, limit=limit)))
                    if reranker is not None:
                        try:
                            hits = store.hybrid_search(query, tag=tag, limit=limit, reranker=reranker)
                            return SearchResult(query=query, mode="hybrid", reranked=True, hits=_hits(hits))
                        except ValueError:
                            raise
                        except Exception as e:  # noqa: BLE001 - re-ranker unavailable: keep fused order
                            log.warning("re-ranking failed, using fused order: %s", e)
                            notice = f"Re-ranker unavailable ({type(e).__name__}); results are not re-ranked."
                    hits = store.hybrid_search(query, tag=tag, limit=limit)
                    return SearchResult(query=query, mode="hybrid", hits=_hits(hits), notice=notice)
                except ValueError:
                    raise
                except Exception as e:  # noqa: BLE001 - embedding model unavailable: degrade to keyword, say so
                    log.warning("%s search failed, falling back to keyword: %s", mode, e)
                    notice = f"{mode.capitalize()} search unavailable ({type(e).__name__}); showing keyword results."
            elif mode != "keyword":
                notice = "Semantic search is disabled on this server; showing keyword results."
            return SearchResult(query=query, mode="keyword", hits=_hits(store.search(query, tag=tag, limit=limit)),
                                notice=notice)
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(title="Find similar notes", annotations=READ)
    def find_similar(
        note_id: Annotated[int, Field(ge=1)],
        limit: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> SimilarResult:
        """Notes closest in meaning to the given note, most similar first."""
        if store.embedder is None:
            raise ToolError("Semantic search is disabled on this server.")
        try:
            return SimilarResult(note_id=note_id, similar=_hits(store.similar(note_id, limit=limit)))
        except NoteNotFound as e:
            raise _not_found(e) from e

    @mcp.tool(title="Get note", annotations=READ)
    def get_note(note_id: Annotated[int, Field(ge=1)]) -> NoteOut:
        """Fetch one note or bookmark in full."""
        try:
            return NoteOut.of(store.get(note_id))
        except NoteNotFound as e:
            raise _not_found(e) from e

    @mcp.tool(title="List notes", annotations=READ)
    def list_notes(
        tag: Annotated[str | None, Field(description="Only notes with this tag")] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> NotePage:
        """List notes, most recently updated first. Paginate with offset / next_offset."""
        try:
            notes, total = store.list(tag=tag, limit=limit, offset=offset)
        except ValueError as e:
            raise ToolError(str(e)) from e
        nxt = offset + len(notes)
        return NotePage(notes=[NoteOut.of(n) for n in notes], total=total, next_offset=nxt if nxt < total else None)

    @mcp.tool(title="List tags", annotations=READ)
    def list_tags() -> list[TagCount]:
        """All tags in use, with how many notes carry each."""
        return [TagCount(tag=t, count=n) for t, n in store.tags()]

    # ----------------------------------------------------------------- tools: write

    @mcp.tool(title="Add note", annotations=WRITE)
    def add_note(
        title: Annotated[str, Field(min_length=1, max_length=200)],
        body: Annotated[str, Field(description="Markdown text", max_length=50_000)] = "",
        tags: Tags | None = None,
    ) -> NoteOut:
        """Save a new note."""
        try:
            return NoteOut.of(store.add(title, body, tags))
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(title="Save bookmark", annotations=WRITE)
    def save_bookmark(
        url: Annotated[str, Field(description="http(s) URL to save")],
        title: Annotated[str, Field(min_length=1, max_length=200)],
        notes: Annotated[str, Field(description="Why it's worth keeping", max_length=50_000)] = "",
        tags: Tags | None = None,
    ) -> NoteOut:
        """Save a link with a title and optional notes."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ToolError(f"Not a valid http(s) URL: {url!r}")
        try:
            return NoteOut.of(store.add(title, notes, tags, url=url))
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(title="Update note", annotations=UPDATE)
    def update_note(
        note_id: Annotated[int, Field(ge=1)],
        title: Annotated[str | None, Field(max_length=200)] = None,
        body: Annotated[str | None, Field(description="Replaces the whole body", max_length=50_000)] = None,
        tags: Annotated[list[str] | None, Field(description="Replaces the whole tag set")] = None,
    ) -> NoteOut:
        """Change a note's title, body or tags. Fields left out stay as they are."""
        try:
            return NoteOut.of(store.update(note_id, title=title, body=body, tags=tags))
        except NoteNotFound as e:
            raise _not_found(e) from e
        except ValueError as e:
            raise ToolError(str(e)) from e

    @mcp.tool(title="Delete note", annotations=DELETE)
    def delete_note(note_id: Annotated[int, Field(ge=1)]) -> Deleted:
        """Permanently delete a note or bookmark. Confirm with the user first."""
        try:
            n = store.delete(note_id)
        except NoteNotFound as e:
            raise _not_found(e) from e
        return Deleted(id=n.id, title=n.title)

    # ----------------------------------------------------------------- tools: ingestion

    @mcp.tool(title="Import a web page", annotations=IMPORT)
    def import_url(
        url: Annotated[str, Field(description="Public http(s) page to save, e.g. an article or docs page")],
        tags: Tags | None = None,
    ) -> ImportResult:
        """Fetch a web page, keep its readable text, and save it as a note (searchable like any other).

        Importing the same URL again refreshes that note instead of creating a duplicate.
        """
        report = ingest_url(store, url, tags, fetch=fetch)
        if report.counts["failed"]:
            raise ToolError(f"Could not import {url}: {report.errors[0].split(': ', 1)[-1]}")
        outcome = next(k for k in ("added", "updated", "unchanged", "linked") if report.counts[k])
        return ImportResult(outcome=outcome, note=NoteOut.of(store.get(report.note_ids[0])))

    @mcp.tool(title="Import history", annotations=READ)
    def import_history(limit: Annotated[int, Field(ge=1, le=50)] = 10) -> list[ImportRun]:
        """Recent import runs (folders, bookmark files, web pages) with what each added or changed."""
        return [ImportRun(**r) for r in store.recent_runs(limit)]

    # ----------------------------------------------------------------- tools: memory

    mem = store.memory

    def _memory_error(e: MemoryNotFound) -> ToolError:
        return ToolError(f"No memory with id {e.memory_id}. Use recall or list_memories to find it.")

    @mcp.tool(title="Remember", annotations=WRITE)
    def remember(
        content: Annotated[str, Field(min_length=3, max_length=500,
                                      description="One short, self-contained sentence, e.g. 'Prefers Python over R'")],
        kind: Annotated[MemoryKind, Field(description="preference, fact, project, decision or event")] = "fact",
        importance: Annotated[int, Field(ge=1, le=5, description="1 minor detail ... 5 essential context")] = 3,
        supersedes: Annotated[int | None, Field(description="Id of a memory this one replaces (fact changed)")] = None,
        force: Annotated[bool, Field(description="Save even though similar memories exist (after review)")] = False,
    ) -> RememberResult:
        """Keep something about the user for future conversations.

        If similar memories already exist, nothing is saved and they are returned for you to judge.
        """
        try:
            outcome, m, similar = mem.remember(content, kind, importance, supersedes, force)
        except MemoryNotFound as e:
            raise _memory_error(e) from e
        except ValueError as e:
            raise ToolError(str(e)) from e
        return RememberResult(outcome=outcome, memory=MemoryOut.of(m) if m else None,
                              similar=[MemoryOut.of(x) for x in similar])

    @mcp.tool(title="Recall", annotations=READ)
    def recall(
        query: Annotated[str | None, Field(description="What the conversation is about. Omit to get the most "
                                                      "important recent memories overall.")] = None,
        kinds: Annotated[list[MemoryKind] | None, Field(description="Only these kinds")] = None,
        limit: Annotated[int, Field(ge=1, le=30)] = 8,
    ) -> RecallResult:
        """Retrieve memories ranked by relevance to the query, importance and recency."""
        hits = mem.recall(query, kinds, limit)
        return RecallResult(query=query, memories=[
            RecalledMemory(**MemoryOut.of(h.memory).model_dump(), score=h.score, relevance=h.relevance,
                           recency=h.recency) for h in hits
        ])

    @mcp.tool(title="List memories", annotations=READ)
    def list_memories(
        kinds: Annotated[list[MemoryKind] | None, Field(description="Only these kinds")] = None,
        include_superseded: Annotated[bool, Field(description="Also show replaced versions")] = False,
    ) -> list[MemoryOut]:
        """Every memory, newest first. Use this when the user wants to see everything you keep."""
        return [MemoryOut.of(m) for m in mem.list(kinds, include_superseded)]

    @mcp.tool(title="Review memories", annotations=READ)
    def review_memories() -> MemoryReview:
        """Find possible duplicates and stale minor memories to tidy up. Changes nothing by itself."""
        r = mem.review()
        return MemoryReview(
            possible_duplicates=[DuplicatePair(a=MemoryOut.of(p["a"]), b=MemoryOut.of(p["b"]),
                                               similarity=p["similarity"]) for p in r["possible_duplicates"]],
            stale=[MemoryOut.of(m) for m in r["stale"]],
        )

    @mcp.tool(title="Forget", annotations=DELETE)
    def forget(memory_id: Annotated[int, Field(ge=1)]) -> Deleted:
        """Permanently delete a memory, including earlier versions it replaced."""
        try:
            m = mem.forget(memory_id)
        except MemoryNotFound as e:
            raise _memory_error(e) from e
        return Deleted(id=m.id, title=m.content)

    # ----------------------------------------------------------------- resources

    @mcp.resource("kb://notes/{note_id}", name="note", title="Note", mime_type="text/markdown")
    def note_resource(note_id: str) -> str:
        """A single note rendered as Markdown."""
        try:
            n = store.get(int(note_id))
        except (ValueError, NoteNotFound) as e:
            raise ResourceNotFoundError(f"No note at kb://notes/{note_id}") from e
        lines = [f"# {n.title}", ""]
        if n.url:
            lines += [f"<{n.url}>", ""]
        if n.tags:
            lines += ["Tags: " + ", ".join(f"`{t}`" for t in n.tags), ""]
        lines += [n.body, "", f"_Updated {n.updated_at}_"]
        return "\n".join(lines).strip() + "\n"

    @mcp.resource("kb://tags", name="tags", title="Tags", mime_type="application/json")
    def tags_resource() -> str:
        """Every tag with its note count."""
        return json.dumps([{"tag": t, "count": n} for t, n in store.tags()])

    @mcp.resource("kb://memories", name="memories", title="What the assistant remembers", mime_type="text/markdown")
    def memories_resource() -> str:
        """All active memories grouped by kind, as Markdown."""
        items = mem.list()
        if not items:
            return "No memories yet.\n"
        lines = ["# What I remember", ""]
        for kind in KINDS:
            group = [m for m in items if m.kind == kind]
            if group:
                lines += [f"## {kind.capitalize()}s", *[f"- [{m.id}] {m.content} (importance {m.importance})"
                                                        for m in group], ""]
        return "\n".join(lines)

    @mcp.resource("kb://imports", name="imports", title="Import history", mime_type="application/json")
    def imports_resource() -> str:
        """The last 20 import runs."""
        return json.dumps(store.recent_runs(20))

    @mcp.resource("kb://stats", name="stats", title="Knowledge base stats", mime_type="application/json")
    def stats_resource() -> str:
        """Counts of notes, bookmarks and tags."""
        return json.dumps(store.stats())

    # ----------------------------------------------------------------- prompts

    @mcp.prompt(title="Summarise a topic")
    def summarize_tag(tag: str) -> list[UserMessage]:
        """Summarise everything saved under one tag."""
        notes, total = store.list(tag=tag, limit=50)
        if not notes:
            return [UserMessage(f"I have no notes tagged '{tag}'. Suggest what I could start collecting on it.")]
        corpus = "\n\n".join(f"## [{n.id}] {n.title}\n{n.url or ''}\n{n.body}".strip() for n in notes)
        more = f" (showing 50 of {total})" if total > 50 else ""
        return [
            UserMessage(
                f"Here are my notes tagged '{tag}'{more}:\n\n{corpus}\n\n"
                "Summarise the key ideas in a few bullet points, cite note ids like [3], "
                "and point out gaps or contradictions."
            )
        ]

    @mcp.prompt(title="Start with memory")
    def start_with_memory(topic: str = "") -> list[UserMessage]:
        """Begin a conversation using what the assistant remembers about the user."""
        about = f" about {topic}" if topic else ""
        return [UserMessage(
            f"Before we start{about}: call the recall tool{' with this topic' if topic else ''} and use what you "
            "remember about me to tailor your help. Briefly tell me which memories you are relying on, and "
            "remember anything new and durable I tell you during this conversation."
        )]

    @mcp.prompt(title="Find related notes")
    def find_related(note_id: str) -> list[UserMessage]:
        """Ask the model to find notes connected to one note, using search_notes."""
        try:
            n = store.get(int(note_id))
        except (ValueError, NoteNotFound):
            return [UserMessage(f"There is no note {note_id}. List my recent notes so I can pick one.")]
        return [
            UserMessage(
                f"Here is note [{n.id}] '{n.title}':\n\n{n.body}\n\n"
                "Use the find_similar tool on this note, and search_notes with a couple of queries for its "
                "key ideas, to find related notes. Then explain how each one connects. Cite note ids."
            )
        ]

    return mcp


def build_store(db: str, semantic: bool = True) -> KnowledgeStore:
    return KnowledgeStore(db, embedder=FastEmbedEmbedder() if semantic else None)


def main() -> None:
    """Entry point.

    kb-mcp                 serve over stdio (what Claude Desktop runs)
    kb-mcp --http          serve over Streamable HTTP
    kb-mcp reindex         embed any notes missing embeddings (also downloads the model once)
    kb-mcp eval            compare keyword, semantic, hybrid and re-ranked retrieval on the bundled test set
    kb-mcp eval --memory   check memory merge / review thresholds against the real model
    kb-mcp ingest folder PATH [--prune]   import Markdown/text notes (re-run any time; only changes apply)
    kb-mcp ingest bookmarks FILE          import a browser bookmarks export (.html)
    kb-mcp ingest url URL [--tag T]       import one web page
    kb-mcp runs                           show recent import runs
    kb-mcp memories [--export FILE]       list what the assistant remembers (or export it as JSON)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Personal knowledge-base MCP server")
    commands = ["serve", "reindex", "eval", "ingest", "runs", "memories"]
    parser.add_argument("command", nargs="?", default="serve", choices=commands)
    parser.add_argument("target", nargs="*", help="for ingest: folder PATH | bookmarks FILE | url URL")
    parser.add_argument("--prune", action="store_true",
                        help="ingest folder: delete notes whose file was removed (only notes from that folder)")
    parser.add_argument("--tag", action="append", default=[], help="ingest url: tag to add (repeatable)")
    parser.add_argument("--memory", action="store_true", help="eval: check memory merge thresholds instead")
    parser.add_argument("--export", metavar="FILE", help="memories: write every memory (with history) as JSON")
    parser.add_argument("--db", default=os.environ.get("KB_DB_PATH", "~/.kb-mcp/kb.db"), help="SQLite file")
    parser.add_argument("--no-semantic", action="store_true", help="keyword search only (no embedding model)")
    parser.add_argument("--rerank", action="store_true", help="re-order hybrid results with a cross-encoder")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.command == "eval":
        from kb_mcp import evaluate

        evaluate.print_memory_thresholds() if args.memory else evaluate.main()
        return

    store = build_store(args.db, semantic=not args.no_semantic)
    if args.command == "reindex":
        n = store.reindex()
        print(f"embedded {n} note(s) with {store.embedder.name if store.embedder else 'no model'}")
        return
    if args.command == "memories":
        if args.export:
            from pathlib import Path

            Path(args.export).expanduser().write_text(store.memory.export())
            print(f"exported to {args.export}")
            return
        items = store.memory.list()
        for m in items:
            print(f"[{m.id:>3}] {m.kind:<10} imp {m.importance}  {m.content}")
        print(f"{len(items)} active memories")
        return
    if args.command == "runs":
        for r in store.recent_runs(10):
            counts = ", ".join(f"{k} {r[k]}" for k in ("added", "updated", "unchanged", "linked", "pruned", "failed")
                               if r[k])
            print(f"#{r['id']:<4} {r['started_at']}  {r['source_type']:<9} {r['source']}  {counts or 'no changes'}")
        return
    if args.command == "ingest":
        from kb_mcp import ingest

        if len(args.target) != 2 or args.target[0] not in ("folder", "bookmarks", "url"):
            parser.error("usage: kb-mcp ingest {folder PATH | bookmarks FILE | url URL}")
        kind, where = args.target
        if kind == "folder":
            report = ingest.ingest_folder(store, where, prune=args.prune)
        elif kind == "bookmarks":
            report = ingest.ingest_bookmarks(store, where)
        else:
            report = ingest.ingest_url(store, where, args.tag)
        print(report.summary())
        for err in report.errors[:10]:
            print(f"  ! {err}")
        return

    reranker = FastEmbedReranker() if args.rerank and not args.no_semantic else None
    server = create_server(store, reranker=reranker)
    if args.http:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()

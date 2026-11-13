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

from kb_mcp.embeddings import FastEmbedEmbedder
from kb_mcp.store import KnowledgeStore, Note, NoteNotFound, SearchHit

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
A personal knowledge base of notes and bookmarks.
- Use search_notes before answering questions about things the user saved; cite note ids.
  The default (semantic) mode matches by meaning; use mode="keyword" for exact names, codes or quotes.
- Use find_similar to explore notes related to one the user is looking at.
- Use add_note / save_bookmark only when the user asks to save something.
- delete_note is permanent: confirm with the user first.
"""

READ = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
UPDATE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
DELETE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)


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
    mode: Literal["semantic", "keyword"] = Field(description="The search mode actually used")
    hits: list[Hit]
    notice: str | None = Field(None, description="Set when semantic search was unavailable and keyword was used")


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


def create_server(store: KnowledgeStore) -> MCPServer:
    mcp = MCPServer(name="knowledge-base", version="0.2.0", instructions=INSTRUCTIONS)

    # ----------------------------------------------------------------- tools: read

    @mcp.tool(title="Search notes", annotations=READ)
    def search_notes(
        query: Annotated[str, Field(description="Words to look for in titles and bodies", min_length=1)],
        tag: Annotated[str | None, Field(description="Only search notes with this tag")] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        mode: Annotated[
            Literal["semantic", "keyword"],
            Field(description="semantic: match by meaning (default). keyword: exact words, ranked by BM25."),
        ] = "semantic",
    ) -> SearchResult:
        """Search notes and bookmarks, best matches first."""
        notice = None
        try:
            if mode == "semantic" and store.embedder is not None:
                try:
                    return SearchResult(query=query, mode="semantic",
                                        hits=_hits(store.semantic_search(query, tag=tag, limit=limit)))
                except ValueError:
                    raise
                except Exception as e:  # noqa: BLE001 - model unavailable: degrade to keyword, say so
                    log.warning("semantic search failed, falling back to keyword: %s", e)
                    notice = f"Semantic search unavailable ({type(e).__name__}); showing keyword results."
            elif mode == "semantic":
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
    kb-mcp eval            compare keyword vs semantic retrieval on the bundled test set
    """
    import argparse

    parser = argparse.ArgumentParser(description="Personal knowledge-base MCP server")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve", "reindex", "eval"])
    parser.add_argument("--db", default=os.environ.get("KB_DB_PATH", "~/.kb-mcp/kb.db"), help="SQLite file")
    parser.add_argument("--no-semantic", action="store_true", help="keyword search only (no embedding model)")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.command == "eval":
        from kb_mcp.evaluate import main as eval_main

        eval_main()
        return

    store = build_store(args.db, semantic=not args.no_semantic)
    if args.command == "reindex":
        n = store.reindex()
        print(f"embedded {n} note(s) with {store.embedder.name if store.embedder else 'no model'}")
        return

    server = create_server(store)
    if args.http:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()

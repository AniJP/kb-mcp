"""SQLite storage for notes and bookmarks, with full-text search (FTS5).

This module knows nothing about MCP. The server in `server.py` is a thin protocol layer on
top of it, which keeps the storage logic testable on its own and lets later phases (vector
search, ingestion) plug in behind the same interface.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kb_mcp.embeddings import Embedder, chunk_text

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

_SCHEMA = """
create table if not exists notes (
    id          integer primary key autoincrement,
    title       text not null,
    body        text not null default '',
    url         text,
    created_at  text not null,
    updated_at  text not null
);

create table if not exists note_tags (
    note_id  integer not null references notes(id) on delete cascade,
    tag      text not null,
    primary key (note_id, tag)
);
create index if not exists note_tags_tag on note_tags(tag);

-- External-content FTS index over title + body, kept in sync by triggers.
create virtual table if not exists notes_fts using fts5(
    title, body, content='notes', content_rowid='id', tokenize='porter unicode61'
);
create trigger if not exists notes_ai after insert on notes begin
    insert into notes_fts(rowid, title, body) values (new.id, new.title, new.body);
end;
create trigger if not exists notes_ad after delete on notes begin
    insert into notes_fts(notes_fts, rowid, title, body) values ('delete', old.id, old.title, old.body);
end;
-- Phase 2: one row per chunk of a note, with its embedding (float32 bytes) and the model that
-- produced it. content_hash detects notes edited since they were embedded.
create table if not exists note_chunks (
    note_id       integer not null references notes(id) on delete cascade,
    chunk_index   integer not null,
    text          text not null,
    model         text not null,
    content_hash  text not null,
    embedding     blob not null,
    primary key (note_id, chunk_index)
);

create trigger if not exists notes_au after update on notes begin
    insert into notes_fts(notes_fts, rowid, title, body) values ('delete', old.id, old.title, old.body);
    insert into notes_fts(rowid, title, body) values (new.id, new.title, new.body);
end;
"""

_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_/]{0,49}$")


class NoteNotFound(LookupError):
    def __init__(self, note_id: int):
        super().__init__(f"note {note_id} not found")
        self.note_id = note_id


@dataclass(frozen=True)
class Note:
    id: int
    title: str
    body: str
    url: str | None
    tags: list[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SearchHit:
    note: Note
    score: float
    snippet: str


def normalize_tags(tags: list[str] | None) -> list[str]:
    """Lower-case, trim, de-duplicate and validate tags."""
    out: list[str] = []
    for raw in tags or []:
        tag = raw.strip().lower().replace(" ", "-")
        if not tag:
            continue
        if not _TAG_RE.match(tag):
            raise ValueError(f"invalid tag {raw!r}: use letters, digits, '-', '_' or '/', at most 50 characters")
        if tag not in out:
            out.append(tag)
    return sorted(out)


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: every word must match, last word as a prefix.

    Quoting each token means user input can never be parsed as FTS5 syntax (no injection,
    no syntax errors from stray quotes or operators).
    """
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return ""
    quoted = [f'"{t}"' for t in tokens]
    quoted[-1] += "*"
    return " ".join(quoted)


def _trim(text: str, query: str = "", words: int = 30) -> str:
    """A `words`-long window of `text`, placed where it shares the most words with the query."""
    parts = text.split()
    if len(parts) <= words:
        return text
    wanted = set(re.findall(r"\w+", query.lower()))
    start = 0
    if wanted:
        hits = [1 if set(re.findall(r"\w+", p.lower())) & wanted else 0 for p in parts]
        window = sum(hits[:words])
        best = window
        for i in range(1, len(parts) - words + 1):
            window += hits[i + words - 1] - hits[i - 1]
            if window > best:
                best, start = window, i
    out = " ".join(parts[start : start + words])
    return ("… " if start > 0 else "") + out + (" …" if start + words < len(parts) else "")


def _content_hash(title: str, body: str) -> str:
    return hashlib.sha256(f"{title}\n{body}".encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class KnowledgeStore:
    def __init__(self, path: str | Path = ":memory:", embedder: Embedder | None = None):
        self.embedder = embedder
        self._vectors: tuple[np.ndarray, np.ndarray, list[str]] | None = None  # (note_ids, matrix, texts)
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("pragma foreign_keys = on")
        if self.path != ":memory:":
            self._conn.execute("pragma journal_mode = wal")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"pragma user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    # ------------------------------------------------------------------ reads

    def _tags_for(self, ids: list[int]) -> dict[int, list[str]]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"select note_id, tag from note_tags where note_id in ({marks}) order by tag", ids
        ).fetchall()
        out: dict[int, list[str]] = {i: [] for i in ids}
        for r in rows:
            out[r["note_id"]].append(r["tag"])
        return out

    def _to_notes(self, rows: list[sqlite3.Row]) -> list[Note]:
        tags = self._tags_for([r["id"] for r in rows])
        return [
            Note(r["id"], r["title"], r["body"], r["url"], tags[r["id"]], r["created_at"], r["updated_at"])
            for r in rows
        ]

    def get(self, note_id: int) -> Note:
        row = self._conn.execute("select * from notes where id = ?", (note_id,)).fetchone()
        if row is None:
            raise NoteNotFound(note_id)
        return self._to_notes([row])[0]

    def list(self, tag: str | None = None, limit: int = 20, offset: int = 0) -> tuple[list[Note], int]:
        """Most recently updated first. Returns (page, total matching)."""
        if tag:
            where, args = "where id in (select note_id from note_tags where tag = ?)", [normalize_tags([tag])[0]]
        else:
            where, args = "", []
        total = self._conn.execute(f"select count(*) from notes {where}", args).fetchone()[0]
        rows = self._conn.execute(
            f"select * from notes {where} order by updated_at desc, id desc limit ? offset ?", [*args, limit, offset]
        ).fetchall()
        return self._to_notes(rows), total

    def search(self, query: str, tag: str | None = None, limit: int = 10) -> list[SearchHit]:
        """Keyword search ranked by BM25 (title matches weigh 3x body matches)."""
        fts = _fts_query(query)
        if not fts:
            return []
        args: list[object] = [fts]
        tag_filter = ""
        if tag:
            tag_filter = "and n.id in (select note_id from note_tags where tag = ?)"
            args.append(normalize_tags([tag])[0])
        args.append(limit)
        rows = self._conn.execute(
            f"""
            select n.*, bm25(notes_fts, 3.0, 1.0) as rank,
                   snippet(notes_fts, 1, '[', ']', ' … ', 12) as snip
            from notes_fts join notes n on n.id = notes_fts.rowid
            where notes_fts match ? {tag_filter}
            order by rank
            limit ?
            """,
            args,
        ).fetchall()
        notes = self._to_notes(rows)
        # bm25() is lower-is-better and negative; expose a positive higher-is-better score.
        return [SearchHit(n, round(-r["rank"], 4), r["snip"] or n.title) for n, r in zip(notes, rows, strict=True)]

    def tags(self) -> list[tuple[str, int]]:
        rows = self._conn.execute(
            "select tag, count(*) as n from note_tags group by tag order by n desc, tag"
        ).fetchall()
        return [(r["tag"], r["n"]) for r in rows]

    def stats(self) -> dict[str, int]:
        c = self._conn
        return {
            "notes": c.execute("select count(*) from notes where url is null").fetchone()[0],
            "bookmarks": c.execute("select count(*) from notes where url is not null").fetchone()[0],
            "tags": c.execute("select count(distinct tag) from note_tags").fetchone()[0],
            "embedded_chunks": c.execute("select count(*) from note_chunks").fetchone()[0],
        }

    # ------------------------------------------------------------------ writes

    def add(self, title: str, body: str = "", tags: list[str] | None = None, url: str | None = None) -> Note:
        title = title.strip()
        if not title:
            raise ValueError("title must not be empty")
        clean_tags = normalize_tags(tags)
        now = _now()
        with self._tx() as c:
            cur = c.execute(
                "insert into notes (title, body, url, created_at, updated_at) values (?, ?, ?, ?, ?)",
                (title, body, url, now, now),
            )
            note_id = cur.lastrowid
            c.executemany("insert into note_tags values (?, ?)", [(note_id, t) for t in clean_tags])
        self._embed_quietly(note_id, title, body)
        return self.get(note_id)

    def update(
        self,
        note_id: int,
        title: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
        url: str | None = None,
    ) -> Note:
        """Change only the fields given; `tags` replaces the whole tag set when provided."""
        current = self.get(note_id)
        new_title = current.title if title is None else title.strip()
        if not new_title:
            raise ValueError("title must not be empty")
        with self._tx() as c:
            c.execute(
                "update notes set title = ?, body = ?, url = ?, updated_at = ? where id = ?",
                (
                    new_title,
                    current.body if body is None else body,
                    current.url if url is None else url,
                    _now(),
                    note_id,
                ),
            )
            if tags is not None:
                c.execute("delete from note_tags where note_id = ?", (note_id,))
                c.executemany("insert into note_tags values (?, ?)", [(note_id, t) for t in normalize_tags(tags)])
        updated = self.get(note_id)
        self._embed_quietly(note_id, updated.title, updated.body)
        return updated

    def delete(self, note_id: int) -> Note:
        note = self.get(note_id)
        with self._tx() as c:
            c.execute("delete from notes where id = ?", (note_id,))
        self._vectors = None
        return note

    # ------------------------------------------------------------------ embeddings (Phase 2)

    def _embed_note(self, note_id: int, title: str, body: str) -> None:
        assert self.embedder is not None
        chunks = chunk_text(title, body)
        vectors = self.embedder.embed_documents(chunks)
        digest = _content_hash(title, body)
        with self._tx() as c:
            c.execute("delete from note_chunks where note_id = ?", (note_id,))
            c.executemany(
                "insert into note_chunks values (?, ?, ?, ?, ?, ?)",
                [
                    (note_id, i, text, self.embedder.name, digest, vec.astype(np.float32).tobytes())
                    for i, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
        self._vectors = None

    def _embed_quietly(self, note_id: int, title: str, body: str) -> None:
        """Embed on write, but never let a missing model (e.g. offline) block saving a note.

        A note that fails here is simply stale and gets embedded by the next reindex().
        """
        if self.embedder is None:
            return
        try:
            self._embed_note(note_id, title, body)
        except Exception as e:  # noqa: BLE001 - any model/runtime failure must not lose the user's write
            log.warning("embedding note %s failed, will retry on reindex: %s", note_id, e)

    def stale_note_ids(self) -> list[int]:
        """Notes with no embeddings from the current model, or edited since they were embedded."""
        if self.embedder is None:
            return []
        rows = self._conn.execute(
            """
            select n.id, n.title, n.body,
                   (select content_hash from note_chunks c where c.note_id = n.id and c.model = ? limit 1) as h
            from notes n
            """,
            (self.embedder.name,),
        ).fetchall()
        return [r["id"] for r in rows if r["h"] != _content_hash(r["title"], r["body"])]

    def reindex(self) -> int:
        """Embed every stale note. Cheap when nothing changed. Returns how many notes were embedded."""
        stale = self.stale_note_ids()
        for note_id in stale:
            n = self.get(note_id)
            self._embed_note(n.id, n.title, n.body)
        return len(stale)

    def _load_vectors(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if self._vectors is None:
            assert self.embedder is not None
            rows = self._conn.execute(
                "select note_id, text, embedding from note_chunks where model = ? order by note_id, chunk_index",
                (self.embedder.name,),
            ).fetchall()
            if rows:
                ids = np.array([r["note_id"] for r in rows])
                matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            else:
                ids, matrix = np.zeros(0, dtype=int), np.zeros((0, 1), dtype=np.float32)
            self._vectors = (ids, matrix, [r["text"] for r in rows])
        return self._vectors

    def _rank_notes(self, sims: np.ndarray, ids: np.ndarray, texts: list[str],
                    allowed: set[int] | None, exclude: int | None, limit: int, query: str = "") -> list[SearchHit]:
        """Best-scoring chunk per note, highest first."""
        best: dict[int, tuple[float, str]] = {}
        for i in np.argsort(-sims):
            nid = int(ids[i])
            if nid == exclude or (allowed is not None and nid not in allowed) or nid in best:
                continue
            best[nid] = (float(sims[i]), texts[i])
            if len(best) == limit:
                break
        notes = {n.id: n for n in self._to_notes(
            self._conn.execute(
                f"select * from notes where id in ({','.join('?' * len(best))})", list(best)
            ).fetchall()
        )} if best else {}
        return [
            SearchHit(notes[nid], round(score, 4), _trim(text.split("\n", 1)[-1], query))
            for nid, (score, text) in best.items()
        ]

    def _ids_with_tag(self, tag: str | None) -> set[int] | None:
        if not tag:
            return None
        t = normalize_tags([tag])[0]
        return {r[0] for r in self._conn.execute("select note_id from note_tags where tag = ?", (t,))}

    def semantic_search(self, query: str, tag: str | None = None, limit: int = 10) -> list[SearchHit]:
        """Rank notes by cosine similarity between the query and each note's best chunk."""
        if self.embedder is None:
            raise RuntimeError("semantic search needs an embedder")
        if not query.strip():
            return []
        self.reindex()
        ids, matrix, texts = self._load_vectors()
        if len(ids) == 0:
            return []
        sims = matrix @ self.embedder.embed_query(query)
        return self._rank_notes(sims, ids, texts, self._ids_with_tag(tag), None, limit, query)

    def similar(self, note_id: int, limit: int = 5) -> list[SearchHit]:
        """Notes closest in meaning to `note_id` (its chunks averaged into one vector)."""
        if self.embedder is None:
            raise RuntimeError("similarity needs an embedder")
        self.get(note_id)  # raises NoteNotFound
        self.reindex()
        ids, matrix, texts = self._load_vectors()
        own = matrix[ids == note_id]
        if len(own) == 0 or len(ids) == len(own):
            return []
        centre = own.mean(axis=0)
        centre /= np.linalg.norm(centre) or 1.0
        return self._rank_notes(matrix @ centre, ids, texts, None, note_id, limit)

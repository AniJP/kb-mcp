"""SQLite storage for notes and bookmarks, with full-text search (FTS5).

This module knows nothing about MCP. The server in `server.py` is a thin protocol layer on
top of it, which keeps the storage logic testable on its own and lets later phases (vector
search, ingestion) plug in behind the same interface.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

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


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class KnowledgeStore:
    def __init__(self, path: str | Path = ":memory:"):
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
        return self.get(note_id)

    def delete(self, note_id: int) -> Note:
        note = self.get(note_id)
        with self._tx() as c:
            c.execute("delete from notes where id = ?", (note_id,))
        return note

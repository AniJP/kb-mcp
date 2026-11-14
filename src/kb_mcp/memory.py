"""Phase 5: agent memory. Things the assistant learns about the user and keeps on its own.

Notes are documents the user chooses to save. Memories are short facts the assistant picks up in
conversation ("prefers concise answers", "applying for data engineering roles", "decided to use
DuckDB over Postgres for the NBA project") so the next conversation starts with that context.

Retrieval follows the recency x importance x relevance idea from "Generative Agents" (Park et
al., 2023): a memory surfaces when it is relevant to the current query, important, and recent.

    score = W_RELEVANCE * relevance + W_IMPORTANCE * importance/5 + W_RECENCY * 0.5 ** (age_days / half_life)

Memories are never silently rewritten, and embeddings never decide what is "the same". Measured
on the real model (`kb-mcp eval --memory`), paraphrases of one fact score 0.78-0.88 cosine while
contradictory facts ("Lives in Virginia" / "Lives in Maryland") score 0.80-0.83: no threshold
separates them. So:

  * identical text (ignoring case and spacing) of the same kind is merged automatically;
  * anything similar (cosine >= SIMILAR_THRESHOLD) is NOT saved: `remember` returns
    "needs_review" with the similar memories, and the assistant, which understands the sentences,
    decides: same fact (do nothing), changed fact (`supersedes=<id>`), or genuinely new (`force`);
  * a changed fact replaces the old one through `supersedes`, which keeps the old version as
    history but hides it from recall.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np

from kb_mcp.embeddings import Embedder

KINDS = ("preference", "fact", "project", "decision", "event")

W_RELEVANCE, W_IMPORTANCE, W_RECENCY = 0.6, 0.25, 0.15
HALF_LIFE_DAYS = 30.0
SIMILAR_THRESHOLD = 0.75  # cosine at or above which an existing memory is shown for review
WORD_OVERLAP_THRESHOLD = 0.6  # the same check when no embedding model is available

_SCHEMA = """
create table if not exists memories (
    id             integer primary key autoincrement,
    content        text not null,
    kind           text not null,
    importance     integer not null check (importance between 1 and 5),
    created_at     text not null,
    updated_at     text not null,
    superseded_by  integer references memories(id) on delete set null,
    model          text,
    embedding      blob
);
create index if not exists memories_active on memories(superseded_by);
"""


class MemoryNotFound(LookupError):
    def __init__(self, memory_id: int):
        super().__init__(f"memory {memory_id} not found")
        self.memory_id = memory_id


@dataclass(frozen=True)
class Memory:
    id: int
    content: str
    kind: str
    importance: int
    created_at: str
    updated_at: str
    superseded_by: int | None


@dataclass(frozen=True)
class Recalled:
    memory: Memory
    score: float
    relevance: float
    recency: float


def _now() -> datetime:
    return datetime.now(UTC)


def _age_days(ts: str, now: datetime) -> float:
    return max((now - datetime.fromisoformat(ts)).total_seconds() / 86400, 0.0)


def recency_weight(age_days: float, half_life: float = HALF_LIFE_DAYS) -> float:
    return 0.5 ** (age_days / half_life)


def score(relevance: float, importance: int, age_days: float) -> float:
    return W_RELEVANCE * relevance + W_IMPORTANCE * importance / 5 + W_RECENCY * recency_weight(age_days)


class MemoryStore:
    """Lives in the same SQLite file as the notes, with its own table and lifecycle."""

    def __init__(self, conn: sqlite3.Connection, embedder: Embedder | None = None, clock=_now):
        self._conn = conn
        self.embedder = embedder
        self._clock = clock
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ helpers

    def _row(self, r: sqlite3.Row) -> Memory:
        return Memory(r["id"], r["content"], r["kind"], r["importance"], r["created_at"], r["updated_at"],
                      r["superseded_by"])

    def _embed(self, text: str) -> np.ndarray | None:
        if self.embedder is None:
            return None
        try:
            return self.embedder.embed_documents([text])[0]
        except Exception:  # noqa: BLE001 - memory must still save without a model
            return None

    def _active_rows(self, kinds: list[str] | None = None) -> list[sqlite3.Row]:
        sql, args = "select * from memories where superseded_by is null", []
        if kinds:
            sql += f" and kind in ({','.join('?' * len(kinds))})"
            args = list(kinds)
        return self._conn.execute(sql, args).fetchall()

    def _vectors(self, rows: list[sqlite3.Row]) -> tuple[list[sqlite3.Row], np.ndarray]:
        """Rows that have an embedding from the current model, and their matrix."""
        if self.embedder is None:
            return [], np.zeros((0, 1), dtype=np.float32)
        keep = [r for r in rows if r["embedding"] is not None and r["model"] == self.embedder.name]
        if not keep:
            return [], np.zeros((0, 1), dtype=np.float32)
        return keep, np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in keep])

    def get(self, memory_id: int) -> Memory:
        r = self._conn.execute("select * from memories where id = ?", (memory_id,)).fetchone()
        if r is None:
            raise MemoryNotFound(memory_id)
        return self._row(r)

    # ------------------------------------------------------------------ writes

    def remember(self, content: str, kind: str = "fact", importance: int = 3,
                 supersedes: int | None = None, force: bool = False) -> tuple[str, Memory | None, list[Memory]]:
        """Store a memory. Returns (outcome, memory, similar).

        outcome:
          created       saved as a new memory
          merged        identical to an existing memory of the same kind; that one was refreshed
          superseded    saved, replacing memory `supersedes`
          needs_review  NOT saved: `similar` lists existing memories that may say the same thing
                        or contradict it. Call again with supersedes=<id> or force=True.
        """
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            raise ValueError("memory content must not be empty")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        if not 1 <= importance <= 5:
            raise ValueError("importance must be between 1 and 5")
        if supersedes is not None:
            old = self.get(supersedes)
            if old.superseded_by is not None:
                raise ValueError(f"memory {supersedes} was already replaced by memory {old.superseded_by}")

        vec = self._embed(content)
        now = self._clock().isoformat(timespec="seconds")

        if supersedes is None:
            same = self._identical(content, kind)
            if same is not None:
                with self._conn:
                    self._conn.execute(
                        "update memories set importance = max(importance, ?), updated_at = ? where id = ?",
                        (importance, now, same.id),
                    )
                return "merged", self.get(same.id), []
            if not force:
                similar = self.similar_to(content, vec)
                if similar:
                    return "needs_review", None, similar

        with self._conn:
            cur = self._conn.execute(
                "insert into memories (content, kind, importance, created_at, updated_at, model, embedding)"
                " values (?, ?, ?, ?, ?, ?, ?)",
                (content, kind, importance, now, now,
                 self.embedder.name if vec is not None and self.embedder else None,
                 vec.astype(np.float32).tobytes() if vec is not None else None),
            )
            new_id = cur.lastrowid
            if supersedes is not None:
                self._conn.execute("update memories set superseded_by = ? where id = ?", (new_id, supersedes))
        return ("superseded" if supersedes is not None else "created"), self.get(new_id), []

    def _identical(self, content: str, kind: str) -> Memory | None:
        for r in self._active_rows([kind]):
            if re.sub(r"\s+", " ", r["content"]).strip().lower() == content.lower():
                return self._row(r)
        return None

    def similar_to(self, content: str, vec: np.ndarray | None = None, limit: int = 5) -> list[Memory]:
        """Active memories (any kind) that might express the same fact or contradict it."""
        rows = self._active_rows()
        scored: dict[int, float] = {}
        keep, matrix = self._vectors(rows) if vec is not None else ([], None)
        if keep:
            for r, sim in zip(keep, matrix @ vec, strict=True):
                if sim >= SIMILAR_THRESHOLD:
                    scored[r["id"]] = float(sim)
        words = set(re.findall(r"\w+", content.lower()))
        embedded = {r["id"] for r in keep}
        for r in rows:
            if r["id"] in embedded or not words:
                continue
            mw = set(re.findall(r"\w+", r["content"].lower()))
            overlap = len(words & mw) / max(len(words), len(mw))
            if overlap >= WORD_OVERLAP_THRESHOLD:
                scored[r["id"]] = overlap
        by_id = {r["id"]: r for r in rows}
        return [self._row(by_id[i]) for i in sorted(scored, key=lambda i: -scored[i])[:limit]]

    def forget(self, memory_id: int) -> Memory:
        """Delete a memory and its history chain (versions it replaced)."""
        m = self.get(memory_id)
        ids, frontier = {memory_id}, [memory_id]
        while frontier:
            rows = self._conn.execute(
                f"select id from memories where superseded_by in ({','.join('?' * len(frontier))})", frontier
            ).fetchall()
            frontier = [r["id"] for r in rows if r["id"] not in ids]
            ids.update(frontier)
        with self._conn:
            self._conn.execute(f"delete from memories where id in ({','.join('?' * len(ids))})", list(ids))
        return m

    # ------------------------------------------------------------------ reads

    def recall(self, query: str | None = None, kinds: list[str] | None = None, limit: int = 8) -> list[Recalled]:
        """Most useful active memories for `query` (or overall, if no query)."""
        rows = self._active_rows(kinds)
        if not rows:
            return []
        now = self._clock()
        relevance = {r["id"]: 1.0 for r in rows}  # no query: rank on importance and recency alone
        if query and query.strip():
            relevance = {r["id"]: 0.0 for r in rows}
            q_vec = None
            if self.embedder is not None:
                try:
                    q_vec = self.embedder.embed_query(query)
                except Exception:  # noqa: BLE001 - fall back to word overlap
                    q_vec = None
            keep, matrix = self._vectors(rows) if q_vec is not None else ([], None)
            if keep:
                for r, sim in zip(keep, matrix @ q_vec, strict=True):
                    relevance[r["id"]] = max(float(sim), 0.0)
            # Word overlap covers memories saved without a model (and the no-model case).
            words = set(re.findall(r"\w+", query.lower()))
            for r in rows:
                if r["id"] not in {k["id"] for k in keep}:
                    mw = set(re.findall(r"\w+", r["content"].lower()))
                    relevance[r["id"]] = len(words & mw) / len(words) if words else 0.0
        out = []
        for r in rows:
            age = _age_days(r["updated_at"], now)
            rel = relevance[r["id"]]
            out.append(Recalled(self._row(r), round(score(rel, r["importance"], age), 4), round(rel, 4),
                                round(recency_weight(age), 4)))
        out.sort(key=lambda x: -x.score)
        if query and query.strip():
            out = [x for x in out if x.relevance > 0.15] or out[:3]
        return out[:limit]

    def list(self, kinds: list[str] | None = None, include_superseded: bool = False) -> list[Memory]:
        sql = "select * from memories" + ("" if include_superseded else " where superseded_by is null")
        rows = [r for r in self._conn.execute(sql + " order by updated_at desc, id desc").fetchall()
                if not kinds or r["kind"] in kinds]
        return [self._row(r) for r in rows]

    def history(self, memory_id: int) -> list[Memory]:
        """The memory and every earlier version it replaced, newest first."""
        chain, current = [self.get(memory_id)], memory_id
        while True:
            r = self._conn.execute("select * from memories where superseded_by = ?", (current,)).fetchone()
            if r is None:
                return chain
            chain.append(self._row(r))
            current = r["id"]

    def review(self, stale_days: float = 180) -> dict[str, list]:
        """Housekeeping candidates for the user or assistant to act on. Nothing is changed here.

        possible_duplicates: pairs of active memories that say nearly the same thing
        stale: low-importance memories untouched for `stale_days`
        """
        rows = self._active_rows()
        pairs = []
        keep, matrix = self._vectors(rows)
        if len(keep) > 1:
            sims = matrix @ matrix.T
            for i in range(len(keep)):
                for j in range(i + 1, len(keep)):
                    if sims[i, j] >= SIMILAR_THRESHOLD:
                        pairs.append({"a": self._row(keep[i]), "b": self._row(keep[j]),
                                      "similarity": round(float(sims[i, j]), 3)})
        now = self._clock()
        stale = [self._row(r) for r in rows if r["importance"] <= 2 and _age_days(r["updated_at"], now) >= stale_days]
        return {"possible_duplicates": sorted(pairs, key=lambda p: -p["similarity"]), "stale": stale}

    def export(self) -> str:
        return json.dumps([asdict(m) for m in self.list(include_superseded=True)], indent=2)

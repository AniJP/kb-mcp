"""Phase 4: bring existing knowledge in automatically.

Three sources, one pipeline:

  folder     Markdown / text files (an Obsidian vault, a notes folder). Title from front matter,
             the first heading or the file name; tags from front matter and #hashtags.
  bookmarks  The HTML file every browser exports (Chrome, Safari, Firefox, Edge). Folder names
             become tags.
  web        A single web page, fetched and reduced to its readable text.

Every source yields `Document`s. `run_ingest` turns them into notes through
`KnowledgeStore.upsert_from_source`, which makes the pipeline idempotent: running it again adds
nothing new, re-imports only what changed, and never duplicates a URL you already saved by hand.
Each run is logged with counts and per-item errors; one bad file never stops the rest.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from kb_mcp.store import KnowledgeStore, normalize_tags

MAX_BODY_CHARS = 50_000
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
_TRACKING_PARAMS = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid|ref_src)$", re.I)


@dataclass
class Document:
    source_type: str
    source_id: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    url: str | None = None


@dataclass
class RunReport:
    run_id: int
    source_type: str
    source: str
    counts: dict[str, int]
    errors: list[str]
    note_ids: list[int] = field(default_factory=list)

    def summary(self) -> str:
        c = self.counts
        parts = [f"{k} {c[k]}" for k in ("added", "updated", "unchanged", "linked", "pruned", "failed") if c.get(k)]
        return f"{self.source_type} {self.source}: " + (", ".join(parts) or "nothing to do")


# --------------------------------------------------------------------------- helpers


def normalize_url(url: str) -> str:
    """Canonical form used to detect duplicates: lower-case scheme and host, no fragment,
    no tracking parameters, no trailing slash on the path."""
    p = urlparse(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING_PARAMS.match(k)])
    path = p.path.rstrip("/") if p.path not in ("", "/") else ""
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))


def _safe_tags(raw: Iterable[str]) -> list[str]:
    """Normalise tags, silently dropping ones that can't be valid (imports should not fail on a tag)."""
    out = []
    for t in raw:
        cleaned = re.sub(r"[^a-z0-9\-_/]", "-", t.strip().lower().replace(" ", "-")).strip("-")[:50]
        if cleaned and cleaned[0].isalnum():
            out.append(cleaned)
    return normalize_tags(out)


# --------------------------------------------------------------------------- folder of notes

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_HASHTAG = re.compile(r"(?<![\w#/])#([A-Za-z][\w\-/]{1,49})")


def _parse_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Minimal YAML front-matter reader for the fields we use: title and tags."""
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, object] = {}
    key = None
    for line in m.group(1).splitlines():
        if re.match(r"^\s+-\s+", line) and key == "tags":
            meta.setdefault("tags", []).append(line.split("-", 1)[1].strip().strip("'\""))  # type: ignore[union-attr]
            continue
        kv = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if not kv:
            continue
        key, value = kv.group(1).lower(), kv.group(2).strip()
        if key == "tags":
            items = value.strip("[]")
            meta["tags"] = [t.strip().strip("'\"") for t in re.split(r"[,\s]+", items) if t.strip()] if items else []
        elif key == "title":
            meta["title"] = value.strip("'\"")
    return meta, text[m.end():]


def parse_note_file(path: Path, root: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _parse_front_matter(text)
    title = str(meta.get("title") or "")
    if not title:
        heading = re.search(r"^#\s+(.+)$", body, re.M)
        if heading:
            title = heading.group(1).strip()
            body = body.replace(heading.group(0), "", 1)
        else:
            title = path.stem.replace("-", " ").replace("_", " ").strip()
    tags = list(meta.get("tags") or []) + _HASHTAG.findall(body)  # type: ignore[arg-type]
    rel = path.relative_to(root).as_posix()
    return Document("folder", f"{root.as_posix()}::{rel}", title[:200], body.strip()[:MAX_BODY_CHARS], _safe_tags(tags))


def folder_documents(root: str | Path) -> Iterator[Document | Exception]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    for path in sorted(root.rglob("*")):
        hidden = any(part.startswith(".") for part in path.relative_to(root).parts)  # .obsidian, .git, ...
        if path.suffix.lower() not in TEXT_SUFFIXES or hidden:
            continue
        try:
            yield parse_note_file(path, root)
        except Exception as e:  # noqa: BLE001 - report and keep going
            yield RuntimeError(f"{path.name}: {e}")


# --------------------------------------------------------------------------- browser bookmarks


class _BookmarkParser(HTMLParser):
    """Parses the Netscape bookmark format: nested <DL> lists of <DT><H3>folder and <DT><A>link."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.folders: list[str] = []
        self.pending_folder: str | None = None
        self.in_h3 = False
        self.in_a = False
        self.href: str | None = None
        self.text = ""
        self.links: list[tuple[str, str, list[str]]] = []
        self.folder_depths: list[int] = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "dl":
            self.depth += 1
            if self.pending_folder is not None:
                self.folders.append(self.pending_folder)
                self.folder_depths.append(self.depth)
                self.pending_folder = None
        elif tag == "h3":
            self.in_h3, self.text = True, ""
        elif tag == "a":
            self.in_a, self.text = True, ""
            self.href = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag == "dl":
            if self.folder_depths and self.folder_depths[-1] == self.depth:
                self.folders.pop()
                self.folder_depths.pop()
            self.depth -= 1
        elif tag == "h3" and self.in_h3:
            self.in_h3, self.pending_folder = False, self.text.strip()
        elif tag == "a" and self.in_a:
            self.in_a = False
            if self.href:
                self.links.append((self.href, self.text.strip(), list(self.folders)))

    def handle_data(self, data):
        if self.in_h3 or self.in_a:
            self.text += data


_BROWSER_ROOT_FOLDERS = {"bookmarks bar", "bookmarks toolbar", "favorites bar", "other bookmarks", "bookmarks menu",
                         "mobile bookmarks", "favorites"}


def bookmark_documents(path: str | Path) -> Iterator[Document | Exception]:
    parser = _BookmarkParser()
    parser.feed(Path(path).expanduser().read_text(encoding="utf-8", errors="replace"))
    for href, title, folders in parser.links:
        if urlparse(href).scheme not in ("http", "https"):
            continue  # javascript: bookmarklets, chrome:// pages, place: queries
        url = normalize_url(href)
        tags = [f for f in folders if f.lower() not in _BROWSER_ROOT_FOLDERS]
        body = f"Saved in: {' / '.join(folders)}" if folders else ""
        yield Document("bookmarks", url, (title or url)[:200], body, _safe_tags(tags), url)


# --------------------------------------------------------------------------- web pages


class UnsafeURL(ValueError):
    pass


def check_public_url(url: str, resolve: Callable[[str], list[str]] | None = None) -> None:
    """Refuse anything but http(s) to a public address.

    The import_url tool can be called by the model, so without this a prompt-injected page could
    make the server fetch http://localhost:..., the cloud metadata endpoint or a router admin page
    (server-side request forgery).
    """
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise UnsafeURL(f"only http(s) URLs can be imported: {url!r}")
    resolve = resolve or (lambda host: [ai[4][0] for ai in socket.getaddrinfo(host, None)])
    try:
        addresses = resolve(p.hostname)
    except OSError as e:
        raise UnsafeURL(f"cannot resolve {p.hostname}: {e}") from e
    for addr in addresses:
        ip = ipaddress.ip_address(addr.split("%")[0])
        if not ip.is_global:
            raise UnsafeURL(f"{p.hostname} resolves to a non-public address ({ip}); refusing to fetch")


class _ReadableText(HTMLParser):
    """Keeps the text a person would read; drops scripts, styles, navigation and page chrome."""

    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button", "template"}
    BLOCK = {"p", "div", "section", "article", "main", "li", "tr", "br", "pre", "blockquote",
             "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.title = ""
        self.in_title = False
        self.chunks: list[str] = []
        self.main_start: int | None = None
        self.main_end: int | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True
        elif tag in ("article", "main") and self.main_start is None and not self.skip_depth:
            self.main_start = len(self.chunks)
        if tag in self.BLOCK and not self.skip_depth:
            self.chunks.append("\n")
            if tag in {"h1", "h2", "h3"}:
                self.chunks.append("## ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag == "title":
            self.in_title = False
        elif tag in ("article", "main") and self.main_start is not None and self.main_end is None:
            self.main_end = len(self.chunks)

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip_depth:
            self.chunks.append(data)

    def text(self) -> str:
        chunks = self.chunks
        if self.main_start is not None:
            main = chunks[self.main_start : self.main_end]
            if len("".join(main).split()) >= 50:  # use <article>/<main> only if it holds real content
                chunks = main
        raw = unescape("".join(chunks))
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
        out, blank = [], False
        for ln in lines:
            if ln and ln != "##":
                out.append(ln)
                blank = False
            elif not blank and out:
                out.append("")
                blank = True
        return "\n".join(out).strip()


def extract_page(html: str) -> tuple[str, str]:
    """(title, readable text) from an HTML page."""
    parser = _ReadableText()
    parser.feed(html)
    return re.sub(r"\s+", " ", unescape(parser.title)).strip(), parser.text()


def fetch_html(url: str, max_bytes: int = 2_000_000, timeout: float = 15.0) -> tuple[str, str]:
    """GET a page with a size cap and timeout. Returns (final_url, html). Redirects are re-checked."""
    check_public_url(url)

    class _Recheck(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            check_public_url(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_Recheck)
    req = urllib.request.Request(url, headers={"User-Agent": "kb-mcp/0.4 (+personal knowledge base importer)"})
    with opener.open(req, timeout=timeout) as resp:
        ctype = resp.headers.get_content_type()
        if ctype not in ("text/html", "application/xhtml+xml", "text/plain"):
            raise ValueError(f"not a web page ({ctype})")
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"page is larger than {max_bytes // 1_000_000} MB")
        charset = resp.headers.get_content_charset() or "utf-8"
        html = data.decode(charset, errors="replace")
        if ctype == "text/plain":
            html = f"<pre>{html}</pre>"
        return resp.geturl(), html


def web_document(url: str, tags: list[str] | None = None,
                 fetch: Callable[[str], tuple[str, str]] = fetch_html) -> Document:
    final_url, html = fetch(url)
    title, text = extract_page(html)
    if not text:
        raise ValueError("no readable text found on the page")
    canonical = normalize_url(final_url)
    return Document("web", canonical, (title or canonical)[:200], text[:MAX_BODY_CHARS], _safe_tags(tags or []),
                    canonical)


# --------------------------------------------------------------------------- the pipeline


def run_ingest(
    store: KnowledgeStore,
    source_type: str,
    source: str,
    documents: Iterable[Document | Exception],
    prune: bool = False,
    prune_prefix: str = "",
) -> RunReport:
    """Upsert every document, log the run, and optionally remove notes whose source item vanished.

    `prune` only ever touches notes created from this same source (matched by source_type and
    `prune_prefix`), never hand-written notes or other sources, and is off by default.
    """
    run_id = store.start_run(source_type, source)
    counts = {"added": 0, "updated": 0, "unchanged": 0, "linked": 0, "pruned": 0, "failed": 0}
    errors: list[str] = []
    seen: set[str] = set()
    note_ids: list[int] = []
    for doc in documents:
        if isinstance(doc, Exception):
            counts["failed"] += 1
            errors.append(str(doc))
            continue
        seen.add(doc.source_id)
        try:
            outcome, note_id = store.upsert_from_source(doc.source_type, doc.source_id, doc.title, doc.body,
                                                        doc.tags, doc.url, run_id)
            counts[outcome] += 1
            note_ids.append(note_id)
        except Exception as e:  # noqa: BLE001 - one bad item must not stop the run
            counts["failed"] += 1
            errors.append(f"{doc.source_id}: {e}")
    if prune:
        for source_id, note_id in store.source_ids(source_type, prune_prefix).items():
            if source_id not in seen:
                try:
                    store.delete(note_id)
                    counts["pruned"] += 1
                except Exception as e:  # noqa: BLE001
                    errors.append(f"prune {source_id}: {e}")
    store.finish_run(run_id, counts, errors)
    return RunReport(run_id, source_type, source, counts, errors, note_ids)


def ingest_folder(store: KnowledgeStore, path: str | Path, prune: bool = False) -> RunReport:
    root = Path(path).expanduser().resolve()
    return run_ingest(store, "folder", str(root), folder_documents(root), prune=prune,
                      prune_prefix=f"{root.as_posix()}::")


def ingest_bookmarks(store: KnowledgeStore, path: str | Path) -> RunReport:
    return run_ingest(store, "bookmarks", str(Path(path).expanduser()), bookmark_documents(path))


def ingest_url(store: KnowledgeStore, url: str, tags: list[str] | None = None,
               fetch: Callable[[str], tuple[str, str]] = fetch_html) -> RunReport:
    def docs():
        try:
            yield web_document(url, tags, fetch)
        except Exception as e:  # noqa: BLE001
            yield RuntimeError(f"{url}: {e}")

    return run_ingest(store, "web", url, docs())

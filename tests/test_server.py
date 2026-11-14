"""Protocol-level tests: talk to the server through a real MCP client, in-process."""

import json

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from kb_mcp.embeddings import HashingEmbedder, OverlapReranker
from kb_mcp.server import create_server
from kb_mcp.store import KnowledgeStore

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client():
    async with Client(create_server(KnowledgeStore(embedder=HashingEmbedder()))) as c:
        yield c


async def call(client, tool, **args):
    r = await client.call_tool(tool, args)
    assert not r.is_error, r.content[0].text
    return r.structured_content


async def test_tools_are_listed_with_schemas_and_annotations(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "search_notes", "get_note", "list_notes", "list_tags",
        "add_note", "save_bookmark", "update_note", "delete_note", "find_similar",
        "import_url", "import_history",
        "remember", "recall", "list_memories", "review_memories", "forget",
    }
    assert tools["forget"].annotations.destructive_hint is True
    assert tools["recall"].annotations.read_only_hint is True
    assert tools["import_url"].annotations.open_world_hint is True
    for name in ("search_notes", "get_note", "list_notes", "list_tags", "find_similar", "import_history"):
        assert tools[name].annotations.read_only_hint is True
    assert tools["delete_note"].annotations.destructive_hint is True
    assert tools["update_note"].annotations.idempotent_hint is True
    assert tools["add_note"].input_schema["required"] == ["title"]
    assert "hits" in tools["search_notes"].output_schema["properties"]


async def test_add_search_get_flow(client):
    added = await call(client, "add_note", title="Brier score", body="Calibration of probabilities", tags=["ml"])
    assert added["uri"] == f"kb://notes/{added['id']}"
    res = await call(client, "search_notes", query="calibration")
    assert [h["id"] for h in res["hits"]] == [added["id"]]
    got = await call(client, "get_note", note_id=added["id"])
    assert got["tags"] == ["ml"]


async def test_bookmark_url_is_validated(client):
    r = await client.call_tool("save_bookmark", {"url": "javascript:alert(1)", "title": "bad"})
    assert r.is_error and "valid http(s) URL" in r.content[0].text
    ok = await call(client, "save_bookmark", url="https://duckdb.org", title="DuckDB", tags=["db"])
    assert ok["url"] == "https://duckdb.org"


async def test_unknown_id_is_a_tool_error_not_a_crash(client):
    for tool in ("get_note", "update_note", "delete_note"):
        r = await client.call_tool(tool, {"note_id": 999})
        assert r.is_error and "No note with id 999" in r.content[0].text


async def test_schema_validation_rejects_bad_arguments(client):
    r = await client.call_tool("search_notes", {"query": "x", "limit": 500})
    assert r.is_error


async def test_update_and_delete(client):
    n = await call(client, "add_note", title="Draft", tags=["a"])
    u = await call(client, "update_note", note_id=n["id"], title="Final", tags=["b"])
    assert (u["title"], u["tags"]) == ("Final", ["b"])
    d = await call(client, "delete_note", note_id=n["id"])
    assert d == {"id": n["id"], "title": "Final", "deleted": True}
    assert (await call(client, "list_notes"))["total"] == 0


async def test_pagination_contract(client):
    for i in range(5):
        await call(client, "add_note", title=f"n{i}")
    p1 = await call(client, "list_notes", limit=2)
    p3 = await call(client, "list_notes", limit=2, offset=4)
    assert p1["total"] == 5 and p1["next_offset"] == 2
    assert p3["next_offset"] is None and len(p3["notes"]) == 1


async def test_resources(client):
    n = await call(client, "add_note", title="Resource me", body="hello", tags=["t"])
    templates = (await client.list_resource_templates()).resource_templates
    assert [t.uri_template for t in templates] == ["kb://notes/{note_id}"]
    text = (await client.read_resource(f"kb://notes/{n['id']}")).contents[0].text
    assert text.startswith("# Resource me") and "`t`" in text
    stats = json.loads((await client.read_resource("kb://stats")).contents[0].text)
    assert stats == {"notes": 1, "bookmarks": 0, "tags": 1, "embedded_chunks": 1, "memories": 0}
    with pytest.raises(MCPError):
        await client.read_resource("kb://notes/999")


async def test_prompts(client):
    await call(client, "add_note", title="Kafka", body="consumer groups", tags=["streaming"])
    names = {p.name for p in (await client.list_prompts()).prompts}
    assert names == {"summarize_tag", "find_related", "start_with_memory"}
    msg = (await client.get_prompt("summarize_tag", {"tag": "streaming"})).messages[0].content.text
    assert "[1] Kafka" in msg and "cite note ids" in msg


async def test_search_modes(client):
    a = await call(client, "add_note", title="Kafka consumer groups", body="partitions and rebalancing")
    await call(client, "add_note", title="Pasta", body="garlic and olive oil")
    hyb = await call(client, "search_notes", query="kafka partitions")
    assert hyb["mode"] == "hybrid" and hyb["hits"][0]["id"] == a["id"] and hyb["notice"] is None
    assert hyb["reranked"] is False
    sem = await call(client, "search_notes", query="kafka partitions", mode="semantic")
    assert sem["mode"] == "semantic" and sem["hits"][0]["id"] == a["id"]
    kw = await call(client, "search_notes", query="kafka", mode="keyword")
    assert kw["mode"] == "keyword" and [h["id"] for h in kw["hits"]] == [a["id"]]


async def test_find_similar(client):
    a = await call(client, "add_note", title="Kafka consumer groups", body="partitions offsets consumers")
    b = await call(client, "add_note", title="Kafka partitions", body="partitions offsets brokers")
    await call(client, "add_note", title="Pasta", body="garlic olive oil spaghetti")
    res = await call(client, "find_similar", note_id=a["id"])
    ids = [h["id"] for h in res["similar"]]
    assert a["id"] not in ids and ids[0] == b["id"]
    r = await client.call_tool("find_similar", {"note_id": 999})
    assert r.is_error and "No note with id 999" in r.content[0].text


class BrokenEmbedder(HashingEmbedder):
    def embed_query(self, text):
        raise RuntimeError("model unavailable")


async def test_semantic_failure_falls_back_to_keyword_with_notice():
    async with Client(create_server(KnowledgeStore(embedder=BrokenEmbedder()))) as c:
        await call(c, "add_note", title="offline note", body="still searchable")
        res = await call(c, "search_notes", query="offline")
        assert res["mode"] == "keyword" and res["hits"] and "unavailable" in res["notice"]


async def test_semantic_disabled_server():
    async with Client(create_server(KnowledgeStore())) as c:
        await call(c, "add_note", title="plain")
        res = await call(c, "search_notes", query="plain")
        assert res["mode"] == "keyword" and "disabled" in res["notice"]
        r = await c.call_tool("find_similar", {"note_id": 1})
        assert r.is_error


async def test_hybrid_with_reranker():
    async with Client(create_server(KnowledgeStore(embedder=HashingEmbedder()), reranker=OverlapReranker())) as c:
        await call(c, "add_note", title="Kafka retention", body="retention.ms deletes old segments")
        res = await call(c, "search_notes", query="kafka retention")
        assert res["mode"] == "hybrid" and res["reranked"] is True and res["hits"][0]["title"] == "Kafka retention"


class BrokenReranker(OverlapReranker):
    def score(self, query, documents):
        raise RuntimeError("reranker model missing")


async def test_reranker_failure_keeps_fused_results_with_notice():
    async with Client(create_server(KnowledgeStore(embedder=HashingEmbedder()), reranker=BrokenReranker())) as c:
        await call(c, "add_note", title="still found")
        res = await call(c, "search_notes", query="still found")
        assert res["mode"] == "hybrid" and res["reranked"] is False and res["hits"]
        assert "Re-ranker unavailable" in res["notice"]


PAGE = """<html><head><title>Kafka retention explained</title></head><body>
<nav>Home | Blog | About</nav><article><h1>Kafka retention</h1>
<p>Topics delete old segments after retention.ms whether or not anyone read them.</p>
<p>""" + "Compacted topics keep the latest value per key instead. " * 10 + """</p></article>
<footer>Copyright</footer><script>track()</script></body></html>"""


async def test_import_url_tool_is_idempotent_and_dedups():
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return url, PAGE

    store = KnowledgeStore(embedder=HashingEmbedder())
    async with Client(create_server(store, fetch=fake_fetch)) as c:
        first = await call(c, "import_url", url="https://blog.example.com/kafka?utm_source=x", tags=["kafka"])
        assert first["outcome"] == "added"
        assert first["note"]["title"] == "Kafka retention explained"
        assert first["note"]["url"] == "https://blog.example.com/kafka"
        assert "Copyright" not in first["note"]["body"] and "track()" not in first["note"]["body"]
        again = await call(c, "import_url", url="https://blog.example.com/kafka")
        assert again["outcome"] in ("unchanged", "updated") and again["note"]["id"] == first["note"]["id"]
        found = await call(c, "search_notes", query="retention.ms")
        assert found["hits"][0]["id"] == first["note"]["id"]
        runs = await call(c, "import_history")
        assert len(runs["result"]) == 2 and runs["result"][0]["source_type"] == "web"


async def test_import_url_refuses_private_addresses():
    async with Client(create_server(KnowledgeStore())) as c:
        for url in ("http://localhost:8765/mcp", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd"):
            r = await c.call_tool("import_url", {"url": url})
            assert r.is_error, url

"""Protocol-level tests: talk to the server through a real MCP client, in-process."""

import json

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from kb_mcp.server import create_server
from kb_mcp.store import KnowledgeStore

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client():
    async with Client(create_server(KnowledgeStore())) as c:
        yield c


async def call(client, tool, **args):
    r = await client.call_tool(tool, args)
    assert not r.is_error, r.content[0].text
    return r.structured_content


async def test_tools_are_listed_with_schemas_and_annotations(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "search_notes", "get_note", "list_notes", "list_tags",
        "add_note", "save_bookmark", "update_note", "delete_note",
    }
    for name in ("search_notes", "get_note", "list_notes", "list_tags"):
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
    assert stats == {"notes": 1, "bookmarks": 0, "tags": 1}
    with pytest.raises(MCPError):
        await client.read_resource("kb://notes/999")


async def test_prompts(client):
    await call(client, "add_note", title="Kafka", body="consumer groups", tags=["streaming"])
    names = {p.name for p in (await client.list_prompts()).prompts}
    assert names == {"summarize_tag", "find_related"}
    msg = (await client.get_prompt("summarize_tag", {"tag": "streaming"})).messages[0].content.text
    assert "[1] Kafka" in msg and "cite note ids" in msg

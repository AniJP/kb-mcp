"""End to end over the real stdio transport, the way Claude Desktop launches the server."""

import sys

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

pytestmark = pytest.mark.anyio


async def test_stdio_server_roundtrip(tmp_path):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "kb_mcp.server", "--db", str(tmp_path / "kb.db"), "--no-semantic"]
    )
    async with Client(params) as c:
        r = await c.call_tool("add_note", {"title": "over stdio"})
        assert not r.is_error
        hits = (await c.call_tool("search_notes", {"query": "stdio"})).structured_content["hits"]
        assert hits[0]["title"] == "over stdio"
    assert (tmp_path / "kb.db").exists()

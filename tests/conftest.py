import pytest


@pytest.fixture
def anyio_backend():
    # The MCP SDK is built on anyio; run async tests on asyncio through anyio's pytest plugin.
    return "asyncio"

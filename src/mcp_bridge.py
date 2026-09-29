import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_server_params = StdioServerParameters(
    command="python",
    args=["src/mcp_server.py"],
)

# 缓存 MCP 工具的 schema
_mcp_tools_schema = []
_tool_name_map = {}  # MCP 工具名 -> 原始名


async def _fetch_tools():
    """连接 MCP Server，拉取工具列表，转成 OpenAI 格式"""
    global _mcp_tools_schema
    async with stdio_client(_server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            _mcp_tools_schema = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.input_schema,
                    },
                }
                for t in tools.tools
            ]
    return _mcp_tools_schema


def get_mcp_tools_schema():
    """同步接口：返回 MCP 工具 schema"""
    if not _mcp_tools_schema:
        asyncio.run(_fetch_tools())
    return _mcp_tools_schema


async def _call_tool(name: str, args: dict):
    async with stdio_client(_server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            return result.content[0].text


def call_mcp_tool(name: str, args: dict) -> str:
    """同步接口：调用 MCP 工具"""
    return asyncio.run(_call_tool(name, args))

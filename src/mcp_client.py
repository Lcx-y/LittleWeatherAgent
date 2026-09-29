import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["src/mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 列出工具
            tools = await session.list_tools()
            print("可用工具:")
            for t in tools.tools:
                print(f" - {t.name}: {t.description}")

            # 调用工具
            result = await session.call_tool(
                name="get_weather", arguments={"city": "北京"}
            )
            print("调用结果：", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("weather-server")


@mcp.tool()
def mcp_get_weather(city: str) -> str:
    # 查询实时天气
    if city not in ["北京", "上海", "东京"]:
        return f"不支持查询{city}的天气"
    return f"{city}的天气是晴天，温度25度"


@mcp.tool()
def mcp_calculate(expression: str) -> str:
    # 计算数学表达式
    return str(eval(expression))


if __name__ == "__main__":
    mcp.run()

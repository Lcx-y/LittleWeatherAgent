from openai import OpenAI
import json
import logging
import random
from src.config import client
from src.token_check import count_tokens, compress_history
from src.tool_registry import tool, get_tools_schema, execute_tool
from src.tracer import Tracer
from src.mcp_bridge import get_mcp_tools_schema, call_mcp_tool
import time
import asyncio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ToolTransientError(Exception):
    # 暂时错误，重试可能成功
    pass


class ToolPermanentError(Exception):
    # 永久错误，重试不会成功
    pass


async def _run_one_tool(tc):
    """执行单个工具调用，返回 (tc, result)"""
    args = json.loads(tc.function.arguments)
    logger.info(f"调用工具: {tc.function.name}({args})")
    try:
        # 同步函数丢进线程池，避免阻塞事件循环
        result = await asyncio.to_thread(execute_tool, tc.function.name, args)
    except ValueError:
        # 本地注册表没有，调用 MCP
        try:
            result = await asyncio.to_thread(call_mcp_tool, tc.function.name, args)
        except ToolTransientError as e:
            result = f"工具暂不可用（请重试）: {e}"
            logger.warning(f"工具 {tc.function.name} 暂不可用: {e}")
        except ToolPermanentError as e:
            result = f"工具无法执行（请更换方式）: {e}"
            logger.error(f"工具 {tc.function.name} 无法执行: {e}")
        except Exception as e:
            result = f"工具执行失败: {e}"
            logger.error(f"未知 {tc.function.name} 失败: {e}")
    except ToolTransientError as e:
        result = f"工具暂不可用（请重试）: {e}"
        logger.warning(f"工具 {tc.function.name} 暂不可用: {e}")
    except ToolPermanentError as e:
        result = f"工具无法执行（请更换方式）: {e}"
        logger.error(f"工具 {tc.function.name} 无法执行: {e}")
    except Exception as e:
        result = f"工具执行失败: {e}"
        logger.error(f"未知 {tc.function.name} 失败: {e}")
    return tc, result


async def _run_tools_concurrently(tool_calls):
    """并发执行所有工具调用，返回结果列表"""
    tasks = [_run_one_tool(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)


# ---------- 1. 定义工具（普通 Python 函数） ----------
@tool("查询某个城市实时天气")
def get_weather(city: str) -> str:
    # 模拟随机失败
    if random.random() < 0.3:
        raise ToolTransientError("天气查询暂时不可用")
    # 模拟无法查询
    if city not in ["北京", "上海", "东京"]:
        raise ToolPermanentError(f"无法查询 {city} 的天气")
    # 模拟查天气
    temp = random.randint(15, 30)
    weather = random.choice(["晴", "阴", "雨", "多云"])
    return f"{city}今天{weather}, {temp}度"


@tool("计算数学表达式")
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))


# ---------- 2. 工具描述（告诉模型有哪些工具可用） ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气。必须调用此工具才能获得天气信息，禁止根据历史推断。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，例如 1+2*3",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
]


# ---------- 3. Agent 主循环 ----------
def run_agent(
    user_input: str,
    history: list = None,
    session_id: str = "default",
    max_steps: int = 10,
):
    tracer = Tracer(session_id, user_input)
    messages = []
    # 避免模型“懒惰”，必须调用工具而不是根据历史记录推断
    system_msg = {
        "role": "system",
        "content": "回答天气问题必须调用 get_weather 工具获取实时数据。禁止根据历史对话推断天气，禁止复用之前查询过的城市数据。",
    }
    messages = [system_msg] + messages
    reasoning = ""
    if history:
        messages.extend(history)  # 历史在前
    messages.append({"role": "user", "content": user_input})  # 当前问题在最后

    new_messages = [{"role": "user", "content": user_input}]

    for step in range(max_steps):
        if count_tokens(messages) > 3000:
            logger.info("触发上下文压缩")
            messages = compress_history(messages)
        logger.info(f"第 {step + 1} 步，发送 {len(messages)} 条消息给模型")

        # 合并 schema
        all_tools = get_tools_schema() + get_mcp_tools_schema()

        step_start = time.time()

        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=all_tools,
        )
        step_duration = int((time.time() - step_start) * 1000)
        step_tokens = response.usage.total_tokens if response.usage else 0

        msg = response.choices[0].message

        # 提取当前轮的 reasoning
        current_reasoning = getattr(msg, "reasoning_content", None)
        if current_reasoning is None and hasattr(msg, "model_extra"):
            current_reasoning = msg.model_extra.get("reasoning_content")
        current_reasoning = current_reasoning or ""

        # 情况 A：模型没调工具，直接给答案 → 结束
        if not msg.tool_calls:
            logger.info(f"模型给出最终答案，共 {step + 1} 步")

            tracer.record_step(  # ← 新增
                step=step + 1,
                duration_ms=step_duration,
                tokens=step_tokens,
                response_type="final_answer",
                reasoning=current_reasoning,
                final_answer=msg.content,
                messages_count=len(messages),
            )
            tracer.save(msg.content)

            assistant_msg = {
                "role": "assistant",
                "content": msg.content,
                "reasoning_content": current_reasoning,
            }
            new_messages.append(assistant_msg)
            return msg.content, current_reasoning, new_messages

        # 情况 B：模型要调工具 → 执行，把结果塞回去
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls,
            "reasoning_content": current_reasoning,
        }

        messages.append(assistant_msg)
        new_messages.append(assistant_msg)

        tool_calls_info = [  # ← 新增
            {"name": tc.function.name, "args": json.loads(tc.function.arguments)}
            for tc in msg.tool_calls
        ]
        tracer.record_step(  # ← 新增
            step=step + 1,
            duration_ms=step_duration,
            tokens=step_tokens,
            response_type="tool_call",
            tool_calls=tool_calls_info,
            reasoning=current_reasoning,
            messages_count=len(messages),
        )

        results = asyncio.run(_run_tools_concurrently(msg.tool_calls))

        for tc, result in results:
            args = json.loads(tc.function.arguments)
            logger.info(f"调用工具: {tc.function.name}({args})")
            try:
                result = execute_tool(tc.function.name, args)  # ← 从注册表执行
            except ValueError:
                # 本地注册表没有，调用mcp
                result = call_mcp_tool(tc.function.name, args)

            except ToolTransientError as e:
                result = f"工具暂不可用（请重试）: {e}"
                logger.warning(f"工具 {tc.function.name} 暂不可用: {e}")
            except ToolPermanentError as e:
                result = f"工具无法执行（请更换方式）: {e}"
                logger.error(f"工具 {tc.function.name} 无法执行: {e}")

            except Exception as e:
                result = f"工具执行失败: {e}"
                logger.error(f"未知 {tc.function.name} 失败: {e}")
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            messages.append(tool_msg)
            new_messages.append(tool_msg)

    tracer.save("达到最大步数")
    return "达到最大步数", "未完成", new_messages


if __name__ == "__main__":
    print(run_agent("北京天气怎么样"))
    print("---")
    print(run_agent("北京和东京的平均温度是多少"))

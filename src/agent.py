from openai import OpenAI
import json
import logging
import random
from src.config import client
from src.token_check import count_tokens, compress_history

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


# ---------- 1. 定义工具（普通 Python 函数） ----------
def get_weather(city: str) -> str:
    # 模拟随机失败
    if random.random() < 0.3:
        raise ToolTransientError("天气查询暂时不可用")
    # 模拟无法查询
    if city not in ["北京", "上海", "东京"]:
        raise ToolPermanentError(f"无法查询 {city} 的天气")

    """模拟查天气"""
    return f"{city}今天晴，22度"


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

tool_map = {"get_weather": get_weather, "calculate": calculate}


# ---------- 3. Agent 主循环 ----------
def run_agent(user_input: str, history: list = None, max_steps: int = 10):
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

        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # 提取当前轮的 reasoning
        current_reasoning = getattr(msg, "reasoning_content", None)
        if current_reasoning is None and hasattr(msg, "model_extra"):
            current_reasoning = msg.model_extra.get("reasoning_content")
        current_reasoning = current_reasoning or ""

        # 情况 A：模型没调工具，直接给答案 → 结束
        if not msg.tool_calls:
            logger.info(f"模型给出最终答案，共 {step + 1} 步")

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

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            logger.info(f"调用工具: {tc.function.name}({args})")
            try:
                result = tool_map[tc.function.name](**args)
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

    return "达到最大步数", "未完成", new_messages


if __name__ == "__main__":
    print(run_agent("北京天气怎么样"))
    print("---")
    print(run_agent("北京和东京的平均温度是多少"))

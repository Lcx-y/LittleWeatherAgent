import tiktoken
from src.config import client

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(enc.encode(content))
        if msg.get("tool_calls"):
            total += len(enc.encode(str(msg["tool_calls"])))
    return total


def compress_history(messages: list, keep_recent: int = 6) -> list:
    """保留最近 keep_recent 条消息，把更早的压缩成摘要"""
    if len(messages) <= keep_recent:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # 让模型对旧历史做摘要
    summary_prompt = [
        {
            "role": "system",
            "content": "请用一段话总结以下对话的关键信息：用户的目标、已查到的数据、已确认的结论。只输出摘要，不要额外解释。",
        },
        {"role": "user", "content": str(old_messages)},
    ]
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=summary_prompt,
    )
    summary = response.choices[0].message.content

    # 用摘要替换旧历史
    compressed = [
        {"role": "system", "content": f"以下是之前对话的摘要：{summary}"},
    ]
    compressed.extend(recent_messages)
    return compressed

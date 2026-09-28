from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import run_agent, logger
import json

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    reply: str


import sqlite3

conn = sqlite3.connect("agent.db", check_same_thread=False)
conn.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        reasoning_content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
# 兼容旧表：如果列不存在才添加
cursor = conn.execute("PRAGMA table_info(history)")
columns = [row[1] for row in cursor.fetchall()]
if "reasoning_content" not in columns:
    conn.execute("ALTER TABLE history ADD COLUMN reasoning_content TEXT")
if "tool_call_id" not in columns:
    conn.execute("ALTER TABLE history ADD COLUMN tool_call_id TEXT")
if "tool_calls" not in columns:
    conn.execute("ALTER TABLE history ADD COLUMN tool_calls TEXT")
conn.commit()


def save_message(
    session_id: str,
    role: str,
    content: str,
    reasoning_content=None,
    tool_call_id=None,
    tool_calls=None,
):
    tool_calls_str = None
    if tool_calls is not None:
        tool_calls_str = json.dumps(
            [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls]
        )
    conn.execute(
        "INSERT INTO history (session_id, role, content, reasoning_content, tool_call_id, tool_calls) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, role, content, reasoning_content, tool_call_id, tool_calls_str),
    )
    conn.commit()


def load_history(session_id):
    rows = conn.execute(
        "SELECT role, content, reasoning_content, tool_calls, tool_call_id FROM history WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    result = []
    for role, content, reasoning, tool_call_id, tool_calls_str in rows:
        msg = {"role": role, "content": content or ""}
        if role == "assistant":
            msg["reasoning_content"] = reasoning or ""  # 空字符串也必须传
            if tool_calls_str:
                msg["tool_calls"] = json.loads(tool_calls_str)
        if role == "tool":
            if not tool_call_id:
                continue
            msg["tool_call_id"] = tool_call_id
        result.append(msg)
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    history = load_history(req.session_id)
    print(f"当前session历史条数: {len(history)}")

    # 避免错误中断后，脏历史传入数据库
    try:
        reply, reasoning, new_messages = run_agent(req.message, history=history)
    except Exception as e:
        logger.error("Agent执行失败:{}".format(e), exc_info=True)
        return ChatResponse(reply="Agent执行失败:{}".format(e))

    for m in new_messages:
        if m["role"] == "user":
            save_message(req.session_id, "user", m["content"])
        elif m["role"] == "assistant":
            save_message(
                req.session_id,
                "assistant",
                m.get("content") or "",
                m.get("reasoning_content") or "",
                tool_calls=m.get("tool_calls"),
            )
        elif m["role"] == "tool":
            save_message(
                req.session_id, "tool", m["content"], None, m.get("tool_call_id")
            )

    return ChatResponse(reply=reply)

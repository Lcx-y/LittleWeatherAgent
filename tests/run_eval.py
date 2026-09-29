import json
import sys
import time
from src.agent import run_agent
from src.config import client

with open("tests/eval_set.json", "r", encoding="utf-8") as f:
    cases = json.load(f)


def judge_quality(question, answer):
    """用模型给回答打分，1-5 分"""
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {
                "role": "system",
                "content": "你是评估员。给以下回答打分（1-5），只输出数字。",
            },
            {"role": "user", "content": f"问题：{question}\n回答：{answer}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def evaluate(case):
    """跑一个用例，返回是否通过 + 详情"""
    session_id = f"eval_{case['id']}_{int(time.time())}"
    try:
        reply, reasoning, new_messages = run_agent(case["input"])
    except Exception as e:
        return False, f"执行异常: {e}", []

    # 检查必须包含的字符串
    for keyword in case.get("must_contain", []):
        if keyword not in reply:
            return False, f"缺少关键词: {keyword}", new_messages

    # 检查工具调用
    called_tools = [
        tc.function.name
        for m in new_messages
        if m["role"] == "assistant" and m.get("tool_calls")
        for tc in m["tool_calls"]
    ]
    # expected_tool：列表里任意一个被调用就算通过（取其一）
    if "expected_tool" in case:
        tools = case["expected_tool"]
        if isinstance(tools, str):
            tools = [tools]
        if not any(t in called_tools for t in tools):
            return False, f"未调用工具: {tools}", new_messages

    # expected_tools：每个组"取其一"，组与组之间必须都满足
    if "expected_tools" in case:
        for group in case["expected_tools"]:
            g = group if isinstance(group, list) else [group]
            if not any(t in called_tools for t in g):
                return False, f"缺少工具调用: {g}", new_messages

    return True, "通过", new_messages


def main():
    passed = 0
    failed = 0
    for case in cases:
        ok, msg, _ = evaluate(case)
        status = "✅" if ok else "❌"
        print(f"{status} [{case['id']}] {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n通过: {passed}/{total} ({passed/total*100:.1f}%)")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

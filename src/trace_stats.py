import json
import glob
import sys

files = glob.glob("traces/*.json")
if not files:
    print("没有 trace 文件")
    sys.exit(0)

total_tokens = 0
total_duration = 0
tool_call_counts = {}

for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        trace = json.load(fp)
    total_tokens += trace["total_tokens"]
    total_duration += trace["total_duration_ms"]
    for step in trace["steps"]:
        if step["response_type"] == "tool_call":
            for tc in step["tool_calls"]:
                tool_call_counts[tc["name"]] = tool_call_counts.get(tc["name"], 0) + 1

n = len(files)
print(f"总请求数: {n}")
print(f"总 token: {total_tokens}")
print(f"平均 token/请求: {total_tokens / n:.1f}")
print(f"平均耗时: {total_duration / n:.0f} ms")
print(f"工具调用统计: {tool_call_counts}")

import requests
import time

BASE_URL = "http://127.0.0.1:8000/chat"
session_id = f"test_{int(time.time())}"
print(f"本次测试使用session: {session_id}\n")
print("输入问题开始对话，输入 quit 退出。")
while True:
    msg = input("\n你: ").strip()
    if msg.lower() == "quit":
        break
    if not msg:
        continue
    try:
        resp = requests.post(
            BASE_URL,
            json={"session_id": session_id, "message": msg},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"Agent: {resp.json()['reply']}")
    except requests.exceptions.ConnectionError:
        print("错误：服务未启动，请先运行 run_server.py")
    except Exception as e:
        print(f"错误：{e}")

import json
import os
import time
import uuid
from datetime import datetime

TRACE_DIR = "traces"


class Tracer:
    def __init__(self, session_id: str, user_input: str):
        os.makedirs(TRACE_DIR, exist_ok=True)
        self.request_id = (
            f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        self.session_id = session_id
        self.user_input = user_input
        self.start_time = time.time()
        self.steps = []
        self.total_tokens = 0

    def record_step(
        self,
        step: int,
        duration_ms: int,
        tokens: int,
        response_type: str,
        tool_calls=None,
        reasoning=None,
        final_answer=None,
        messages_count=0,
    ):
        self.steps.append(
            {
                "step": step,
                "duration_ms": duration_ms,
                "tokens": tokens,
                "request_messages_count": messages_count,
                "response_type": response_type,
                "tool_calls": tool_calls,
                "reasoning": reasoning,
                "final_answer": final_answer,
            }
        )
        self.total_tokens += tokens

    def save(self, final_reply: str):
        end_time = time.time()
        trace = {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_input": self.user_input,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "total_duration_ms": int((end_time - self.start_time) * 1000),
            "total_tokens": self.total_tokens,
            "steps": self.steps,
            "final_reply": final_reply,
        }
        path = os.path.join(TRACE_DIR, f"{self.request_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        return path

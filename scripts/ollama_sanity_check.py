"""Validate local Ollama reachability and sequential model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib import error, request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ANALYST_MODEL,
    ENGINEER_MODEL,
    OLLAMA_HOST,
    PLANNER_MODEL,
    REASONING_FALLBACK_MODEL,
    REVIEWER_MODEL,
)


def post(path: str, payload: dict, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{OLLAMA_HOST}{path}",
        method="POST",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def get(path: str, timeout: int = 60) -> dict:
    req = request.Request(f"{OLLAMA_HOST}{path}", method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def generate_once(model: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: ok"}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 8},
    }
    out = post("/api/chat", payload)
    text = out.get("message", {}).get("content", "").strip().lower()
    if "ok" not in text:
        raise RuntimeError(f"Unexpected output for {model}: {text}")
    print(f"{model}: ok")


def main() -> None:
    tags = get("/api/tags")
    models = [m.get("name", "") for m in tags.get("models", [])]
    print(f"Found {len(models)} local model(s).")

    # Required sequential run check for default routing.
    routed_models: list[str] = []
    seen: set[str] = set()
    for model in (
        ANALYST_MODEL,
        PLANNER_MODEL,
        ENGINEER_MODEL,
        REVIEWER_MODEL,
        REASONING_FALLBACK_MODEL,
    ):
        if model not in seen:
            seen.add(model)
            routed_models.append(model)

    for model in routed_models:
        generate_once(model)
    print("Sequential model calls succeeded.")


if __name__ == "__main__":
    main()

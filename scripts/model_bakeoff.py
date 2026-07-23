"""Compare extraction quality and latency across two local Ollama models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib import error, request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ANALYST_MODEL, OLLAMA_HOST, REASONING_FALLBACK_MODEL

DEFAULT_TEXT = """
Evaluate the quality of the following text:
""".strip()

PROMPT = """
Extract structured fields from the section text and return JSON only.
Fields:
- research_question
- methodology
- datasets_or_benchmarks
- variables
- hyperparameters
- evaluation_metrics
""".strip()


def call_ollama(model: str, text: str, timeout: int = 900) -> tuple[dict, float]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 768},
        "format": "json",
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{OLLAMA_HOST}/api/chat",
        method="POST",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Failed calling Ollama model={model}: {exc}") from exc
    elapsed = time.perf_counter() - start
    return result, elapsed


def load_text(path: str | None) -> str:
    if not path:
        return DEFAULT_TEXT
    content = Path(path).read_text(encoding="utf-8").strip()
    return content or DEFAULT_TEXT


def main() -> None:
    parser = argparse.ArgumentParser(description="Model bake-off for extraction quality.")
    parser.add_argument("--text-file", help="Optional plain-text section to analyze.")
    parser.add_argument("--model-a", default=ANALYST_MODEL)
    parser.add_argument("--model-b", default=REASONING_FALLBACK_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    text = load_text(args.text_file)

    for model in (args.model_a, args.model_b):
        print(f"\n=== {model} ===")
        response, elapsed = call_ollama(
            model=model,
            text=text,
            timeout=args.timeout_seconds,
        )
        content = response.get("message", {}).get("content", "").strip()
        print(f"Elapsed: {elapsed:.2f}s")
        print(content if content else "[empty response]")


if __name__ == "__main__":
    main()

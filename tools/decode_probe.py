#!/usr/bin/env python3
"""Single-stream decode speed on three prompts, against the running server (stdlib only).

Streaming, T=0, thinking off; tok/s = (completion tokens - 1) / (time from the first content token to the end): the
time to first token is left out. Median of 3 runs a prompt, after one warm-up request.

    API_URL=http://127.0.0.1:8888 tools/decode_probe.py [label]

The prompts: "structured" counts from 1 to 200 (sparkDash's and bench_decode.py's structured prompt, the most
predictable output there is); "prose" is the hash-map explanation those tools also use; "code (fr)" asks for a real
program in French (a commented binary search tree), the closest of the three to agent work."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

API = os.environ.get("API_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8888')}")
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")
PROBES = [
    ("structured (count 1-200)", "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.", 200),
    ("prose (hash map, en)", "Write a detailed step-by-step explanation of how a hash map works, including collision "
     "handling, resizing, and time complexity. Be thorough.", 200),
    ("code (fr, binary search tree)", "Écris une implémentation Python complète d'un arbre binaire de recherche "
     "(insert, search, delete, parcours in-order), commentée.", 400),
]


def run(prompt: str, max_tokens: int) -> float:
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(API + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    first, tokens = None, 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content") and first is None:
                first = time.time()
    return (tokens - 1) / (time.time() - first)


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else API
    run("Say hello.", 32)
    print(f"{label}: single stream, T=0, thinking off, time to first token excluded, median of 3")
    for name, prompt, n in PROBES:
        v = sorted(run(prompt, n) for _ in range(3))
        print(f"  {name:30s} {v[1]:6.1f} tok/s  (min {v[0]:.1f}, max {v[2]:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Streaming TTFT / prefill / decode battery for live Studio Flash-Next."""
from __future__ import annotations

import os
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone

ENDPOINT = "http://127.0.0.1:8002/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Next-Uncensored-MLX-6bit"
API_KEY = ""
GEN_TOKENS = 256
CELLS = [256, 2048, 8192]
N_WARM = 1
N_MEAS = 3


def make_prompt(target_tokens: int, nonce: str) -> str:
    # "word " is typically 1-2 tokens; we size generously and trust server usage.
    filler = ("alpha bravo charlie delta echo foxtrot gulf hotel india juliet ") * (target_tokens // 10 + 8)
    return (
        f"Nonce {nonce}. After the context, reply with the single letter X, "
        f"then keep repeating the word ping until you stop.\n"
        f"CONTEXT_BEGIN\n{filler}\nCONTEXT_END\nNow output."
    )


def stream(prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json", **({"Authorization":"Bearer "+API_KEY} if API_KEY else {})},
        method="POST",
    )
    t0 = time.perf_counter()
    first = None
    n_chunks = 0
    text = []
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            item = json.loads(line[6:])
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    n_chunks += 1
                    text.append(piece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    t1 = time.perf_counter()
    usage = usage or {}
    prompt_toks = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    comp_toks = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    ttft = None if first is None else first - t0
    decode_s = None if first is None else t1 - first
    prefill_tps = (prompt_toks / ttft) if ttft and ttft > 0 else None
    # first streamed token is the start of decode; remaining generated tokens / decode_s
    decode_tps = None
    if decode_s and decode_s > 0 and comp_toks:
        decode_tps = max(comp_toks - 1, 1) / decode_s
    return {
        "ttft_s": ttft,
        "total_s": t1 - t0,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "chunks": n_chunks,
        "finish": finish,
        "text_head": "".join(text)[:80],
        "text": "".join(text),
        "usage": usage,
    }


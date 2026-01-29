#!/usr/bin/env python3
"""
Memory-safe, deterministic-ish client benchmark for vLLM under tight GPU memory.
- Same client code and parameters for single-process and two-process modes.
- Focuses on end-to-end latency (tail), not peak throughput.
- Uses temperature=0 and stream=false to reduce variance.
Valid conclusions: latency/throughput impact from multi-process scheduling on one GPU.
Out of scope: peak throughput, multi-GPU scaling, or speculative decoding behavior.

Requires: pip install httpx
"""

import argparse
import asyncio
import math
import time
from typing import List, Tuple

import httpx

# Approximate 512-1024 tokens for Qwen2.5 with a long, repetitive prompt.
PROMPT = (
    "You are given a technical brief about systems performance. "
    "Summarize the key constraints, then list the risks, then provide a short plan. "
    "Focus on determinism, memory safety, and reproducibility. "
    "Do not speculate beyond the given facts. "
    "Repeat: Focus on determinism, memory safety, and reproducibility. "
    "The system runs on a single GPU with limited memory. "
    "Avoid excessive buffering and keep the plan minimal. "
    "Latency, especially tail latency, is the main concern. "
    "The workload should be stable and comparable across runs. "
    "The benchmark must be fair and avoid hidden optimizations. "
    "Explain what conclusions are valid and what are out of scope. "
    "Now provide the requested output in three short sections. "
    * 12
)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))
    return s[k]


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
) -> Tuple[float, int]:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = await client.post(f"{base_url}/v1/completions", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    t1 = time.perf_counter()

    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        # Fallback: approximate by whitespace tokens if usage is absent.
        text = (data.get("choices") or [{}])[0].get("text") or ""
        completion_tokens = max(0, len(text.split()))

    return (t1 - t0), int(completion_tokens)


async def run_benchmark(
    base_urls: List[str],
    model: str,
    max_tokens: int,
    total_requests: int,
    concurrency: int,
    timeout_s: float,
) -> Tuple[List[float], List[int], float]:
    sem = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    tokens: List[int] = []

    async with httpx.AsyncClient() as client:

        async def run_one(i: int) -> None:
            base_url = base_urls[i % len(base_urls)]
            async with sem:
                latency, tok = await one_request(
                    client, base_url, model, max_tokens, timeout_s
                )
            latencies.append(latency)
            tokens.append(tok)

        t_start = time.perf_counter()
        tasks = [asyncio.create_task(run_one(i)) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    return latencies, tokens, (t_end - t_start)


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM multi-process overhead benchmark")
    parser.add_argument(
        "--base-urls",
        default="http://127.0.0.1:8000",
        help="Comma-separated base URLs (use one for single-process, two for double).",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--total-requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    base_urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    if not base_urls:
        raise SystemExit("No valid base URLs provided.")
    if args.concurrency < 1 or args.concurrency > 2:
        raise SystemExit("Concurrency must be 1 or 2 to meet benchmark rules.")

    latencies, tokens, elapsed = asyncio.run(
        run_benchmark(
            base_urls,
            args.model,
            args.max_tokens,
            args.total_requests,
            args.concurrency,
            args.timeout_s,
        )
    )

    total_tokens = sum(tokens)
    tps = total_tokens / elapsed if elapsed > 0 else float("nan")
    p50 = percentile(latencies, 0.50)
    p99 = percentile(latencies, 0.99)

    print("=== vLLM multi-process overhead benchmark ===")
    print(f"Base URLs: {', '.join(base_urls)}")
    print(f"Total requests: {args.total_requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Total elapsed wall time (s): {elapsed:.3f}")
    print(f"Total generated tokens: {total_tokens}")
    print(f"Tokens per second: {tps:.3f}")
    print(f"P50 latency (ms): {p50 * 1000:.1f}")
    print(f"P99 latency (ms): {p99 * 1000:.1f}")


if __name__ == "__main__":
    main()

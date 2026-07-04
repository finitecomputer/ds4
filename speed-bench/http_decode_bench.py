#!/usr/bin/env python3
"""Benchmark greedy decode throughput through an OpenAI-compatible HTTP API."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time
import urllib.error
import urllib.request


PROMPTS = {
    "prose160": (
        "Continue exactly this sentence for about 160 words, no markdown: "
        "The archive door opened only after the rain stopped, and inside we found"
    ),
    "integer128": (
        "Continue the sequence for 128 tokens, output only comma-separated integers: "
        "1, 2, 3, 4,"
    ),
    "prose512": (
        "Write one continuous plain-text paragraph of at least 900 words. "
        "No markdown, no headings, no lists, no conclusion. Start exactly with: "
        "The archive door opened only after the rain stopped, and inside we found"
    ),
    "integer512": (
        "Continue this comma-separated integer sequence for at least 900 numbers. "
        "Output only integers and commas, with no explanation and no final sentence: "
        "1, 2, 3, 4,"
    ),
}


def request_completion(args: argparse.Namespace, prompt: str) -> dict[str, object]:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.bearer_token:
        headers["Authorization"] = f"Bearer {args.bearer_token}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{url}: HTTP {exc.code}: {detail}") from exc
    elapsed = time.perf_counter() - started
    obj = json.loads(raw)
    usage = obj.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    content = (
        obj.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if completion_tokens <= 0:
        raise SystemExit(f"{url}: response did not report completion tokens")
    if args.min_completion_tokens and completion_tokens < args.min_completion_tokens:
        raise SystemExit(
            f"{url}: expected at least {args.min_completion_tokens} completion "
            f"tokens, got {completion_tokens}; prompt stopped early"
        )
    return {
        "seconds": elapsed,
        "completion_tokens": completion_tokens,
        "tok_s": completion_tokens / elapsed,
        "preview": content[:160],
    }


def summarize(label: str, samples: list[dict[str, object]]) -> dict[str, object]:
    token_total = sum(int(sample["completion_tokens"]) for sample in samples)
    second_total = sum(float(sample["seconds"]) for sample in samples)
    tok_s_values = [float(sample["tok_s"]) for sample in samples]
    return {
        "label": label,
        "samples": samples,
        "total_completion_tokens": token_total,
        "total_seconds": second_total,
        "aggregate_tok_s": token_total / second_total if second_total > 0 else 0.0,
        "mean_tok_s": statistics.fmean(tok_s_values),
        "min_tok_s": min(tok_s_values),
        "max_tok_s": max(tok_s_values),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    results: dict[str, object] = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "min_completion_tokens": args.min_completion_tokens,
        "warmups_per_prompt": args.warmups,
        "samples_per_prompt": args.samples,
        "prompts": {},
    }
    all_samples: list[dict[str, object]] = []
    for name in args.prompts:
        prompt = PROMPTS[name]
        for warmup_index in range(args.warmups):
            sample = request_completion(args, prompt)
            if not args.quiet:
                print(
                    f"{name} warmup {warmup_index + 1}: "
                    f"{sample['tok_s']:.6f} tok/s "
                    f"({sample['completion_tokens']} tokens in {sample['seconds']:.3f}s)",
                    flush=True,
                )
        samples = []
        for sample_index in range(args.samples):
            sample = request_completion(args, prompt)
            sample["sample_index"] = sample_index + 1
            samples.append(sample)
            all_samples.append(sample)
            if not args.quiet:
                print(
                    f"{name} sample {sample_index + 1}: "
                    f"{sample['tok_s']:.6f} tok/s "
                    f"({sample['completion_tokens']} tokens in {sample['seconds']:.3f}s)",
                    flush=True,
                )
        results["prompts"][name] = summarize(name, samples)
    results["combined"] = summarize("combined", all_samples)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8050/v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--label", default="decode")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--min-completion-tokens",
        type=int,
        default=0,
        help="Fail if any measured response returns fewer completion tokens.",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        choices=sorted(PROMPTS),
        help="Prompt id to run. Repeat to select multiple prompts.",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--bearer-token",
        default=None,
        help="Bearer token for authenticated frontdoors. Defaults to --bearer-token-env when set.",
    )
    parser.add_argument(
        "--bearer-token-env",
        default="",
        help="Read bearer token from this environment variable when --bearer-token is omitted.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.warmups < 0:
        parser.error("--warmups must not be negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.min_completion_tokens < 0:
        parser.error("--min-completion-tokens must not be negative")
    if args.min_completion_tokens > args.max_tokens:
        parser.error("--min-completion-tokens must be <= --max-tokens")
    if not args.bearer_token and args.bearer_token_env:
        args.bearer_token = os.environ.get(args.bearer_token_env)
    if not args.prompts:
        args.prompts = ["prose160", "integer128"]
    result = run(args)
    rendered = json.dumps(result, indent=2)
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fp:
            fp.write(rendered)
            fp.write("\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

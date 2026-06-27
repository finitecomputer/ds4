#!/usr/bin/env python3
"""Validate preserved DS4 DFlash runtime-smoke evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SUMMARY_KEYS = [
    "attempts",
    "drafted",
    "verified",
    "accepted_including_anchor",
    "misses",
    "rejected_draft_tokens",
    "timing_lines",
]

REQUIRED_FILES = [
    "baseline.out",
    "dflash.out",
    "dflash.err",
    "dflash-summary.env",
    "metadata.txt",
    "prompt.txt",
]


def fail(message: str, evidence_dir: Path) -> None:
    print(f"DFlash smoke evidence failed: {message}", file=sys.stderr)
    print(f"Evidence: {evidence_dir}", file=sys.stderr)
    raise SystemExit(1)


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"{path.name}:{lineno}: missing '='")
        key, value = raw.split("=", 1)
        if key in values:
            raise ValueError(f"{path.name}:{lineno}: duplicate key {key}")
        values[key] = value
    return values


def require_uint(values: dict[str, str], key: str, evidence_dir: Path) -> int:
    value = values.get(key)
    if value is None or not value.isdigit():
        fail(f"summary field {key} must be an unsigned integer", evidence_dir)
    return int(value)


def validate(args: argparse.Namespace) -> dict[str, int]:
    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.is_dir():
        fail("evidence directory is missing", evidence_dir)

    for filename in REQUIRED_FILES:
        path = evidence_dir / filename
        if not path.is_file():
            fail(f"missing required artifact {filename}", evidence_dir)

    try:
        metadata = parse_env(evidence_dir / "metadata.txt")
    except ValueError as exc:
        fail(str(exc), evidence_dir)
    if metadata.get("model") != args.model:
        fail("metadata model does not match expected target GGUF", evidence_dir)
    if metadata.get("dflash") != args.dflash:
        fail("metadata dflash does not match expected DFlash artifact", evidence_dir)
    if args.ds4_commit and metadata.get("ds4_commit") != args.ds4_commit:
        fail("metadata ds4_commit does not match expected DS4 DFlash archive commit", evidence_dir)

    if (evidence_dir / "baseline.out").read_bytes() != (evidence_dir / "dflash.out").read_bytes():
        fail("DFlash stdout differs from baseline stdout", evidence_dir)

    try:
        summary_values = parse_env(evidence_dir / "dflash-summary.env")
    except ValueError as exc:
        fail(str(exc), evidence_dir)

    parsed = {key: require_uint(summary_values, key, evidence_dir) for key in SUMMARY_KEYS}

    if args.min_verified < 0:
        fail("min_verified must be an unsigned integer", evidence_dir)
    if parsed["attempts"] <= 0:
        fail("DFlash verifier attempts were not observed", evidence_dir)
    if parsed["drafted"] <= 0:
        fail("DFlash drafted tokens were not observed", evidence_dir)
    if parsed["verified"] < args.min_verified:
        fail("verified DFlash draft tokens are below the required minimum", evidence_dir)
    if parsed["accepted_including_anchor"] < parsed["verified"]:
        fail("accepted-anchor count is below verified draft-token count", evidence_dir)
    if parsed["accepted_including_anchor"] < parsed["attempts"]:
        fail("accepted-anchor count is below verifier attempt count", evidence_dir)
    if parsed["accepted_including_anchor"] > parsed["drafted"] + parsed["attempts"]:
        fail("accepted-anchor count exceeds drafted tokens plus verifier attempts", evidence_dir)
    if parsed["rejected_draft_tokens"] < parsed["misses"]:
        fail("rejected draft-token count is below verifier miss count", evidence_dir)
    if parsed["timing_lines"] <= 0:
        fail("DFlash timing summary was not observed", evidence_dir)

    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dflash", required=True)
    parser.add_argument("--ds4-commit")
    parser.add_argument("--min-verified", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parsed = validate(args)
    for key in SUMMARY_KEYS:
        print(f"{key}={parsed[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

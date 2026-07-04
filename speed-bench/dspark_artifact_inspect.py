#!/usr/bin/env python3
"""Inspect a DeepSeek V4 Flash DSpark draft artifact.

This is intentionally dependency-free so it can run on a Spark before Python ML
packages are installed.  It validates the sharded `mtp.*` layout from
`model.safetensors.index.json` and, when the safetensors files are present,
checks the metadata headers without loading tensor data.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STAGE_RE = re.compile(r"^mtp\.(\d+)\.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len == 0 or header_len > 128 * 1024 * 1024:
            raise ValueError(f"{path}: suspicious safetensors header length {header_len}")
        header = f.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(header)


def stage_for_tensor(name: str) -> int | None:
    match = STAGE_RE.match(name)
    return int(match.group(1)) if match else None


def shape_text(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "missing"
    return f"{meta.get('dtype')} {meta.get('shape')}"


def require(condition: bool, problems: list[str], message: str) -> None:
    if not condition:
        problems.append(message)


def inspect_artifact(artifact_dir: Path, index_name: str) -> int:
    config = load_json(artifact_dir / "config.json")
    index = load_json(artifact_dir / index_name)
    weight_map: dict[str, str] = index.get("weight_map", {})
    problems: list[str] = []

    stage_keys: dict[int, list[str]] = defaultdict(list)
    for name in weight_map:
        stage = stage_for_tensor(name)
        if stage is not None:
            stage_keys[stage].append(name)

    stages = sorted(stage_keys)
    require(stages == list(range(len(stages))), problems, f"MTP stages are not contiguous: {stages}")
    require(config.get("dspark_block_size") == 5, problems, "expected dspark_block_size=5")
    require(config.get("dspark_markov_rank") == 256, problems, "expected dspark_markov_rank=256")
    require(config.get("dspark_target_layer_ids") == [40, 41, 42],
            problems,
            f"unexpected dspark_target_layer_ids={config.get('dspark_target_layer_ids')}")
    require(config.get("sliding_window") == 128, problems, "expected sliding_window=128")

    required_by_name = [
        "mtp.0.main_proj.weight",
        "mtp.0.main_proj.scale",
        "mtp.0.main_norm.weight",
        "mtp.2.norm.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
        "mtp.2.confidence_head.proj.weight",
        "mtp.2.hc_head_fn",
        "mtp.2.hc_head_base",
        "mtp.2.hc_head_scale",
    ]
    shared_base_names = [
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    ]
    for name in required_by_name:
        require(name in weight_map, problems, f"missing required tensor {name}")

    print("DSpark artifact")
    print(f"  dir: {artifact_dir}")
    print(f"  total_size: {index.get('metadata', {}).get('total_size')}")
    print(f"  hidden_size: {config.get('hidden_size')}")
    print(f"  vocab_size: {config.get('vocab_size')}")
    print(f"  block_size: {config.get('dspark_block_size')}")
    print(f"  target_layers: {config.get('dspark_target_layer_ids')}")
    print(f"  markov_rank: {config.get('dspark_markov_rank')}")
    print(f"  sliding_window: {config.get('sliding_window')}")
    print(f"  mtp_stages: {stages}")

    all_headers: dict[str, dict[str, Any]] = {}
    for file_name in sorted(set(weight_map.values())):
        if not file_name.startswith("dspark-mtp-"):
            continue
        path = artifact_dir / file_name
        if path.exists():
            all_headers[file_name] = read_safetensors_header(path)

    print("\nMTP stage files")
    for stage in stages:
        files = sorted({weight_map[name] for name in stage_keys[stage]})
        print(f"  mtp.{stage}: tensors={len(stage_keys[stage])} files={','.join(files)}")

    print("\nShared base tensors")
    for name in shared_base_names:
        file_name = weight_map.get(name)
        print(f"  {name}: {file_name or 'missing'}")
    missing_shared = [name for name in shared_base_names if name not in weight_map]
    require(not missing_shared, problems, f"missing shared base tensors in index: {missing_shared}")

    if all_headers:
        print("\nSafetensors headers")
        for file_name, header in sorted(all_headers.items()):
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            dtypes = Counter(str(v.get("dtype")) for v in tensors.values())
            print(f"  {file_name}: header_tensors={len(tensors)} dtypes={dict(sorted(dtypes.items()))}")
            mapped = {k for k, v in weight_map.items() if v == file_name}
            header_names = set(tensors)
            missing = sorted(mapped - header_names)
            extra = sorted(header_names - mapped)
            require(not missing, problems, f"{file_name}: missing mapped tensors in header: {missing[:5]}")
            require(not extra, problems, f"{file_name}: header has extra tensors: {extra[:5]}")

        print("\nRequired tensor metadata")
        for name in required_by_name:
            file_name = weight_map.get(name)
            meta = all_headers.get(file_name or "", {}).get(name)
            print(f"  {name}: {file_name} {shape_text(meta)}")

        hidden = config.get("hidden_size")
        vocab = config.get("vocab_size")
        markov = config.get("dspark_markov_rank")
        expected_shapes = {
            "mtp.0.main_proj.weight": [hidden, hidden * len(config.get("dspark_target_layer_ids", []))],
            "mtp.2.markov_head.markov_w1.weight": [vocab, markov],
            "mtp.2.markov_head.markov_w2.weight": [vocab, markov],
            "mtp.2.confidence_head.proj.weight": [1, hidden + markov],
            "mtp.2.hc_head_fn": [config.get("hc_mult"), config.get("hc_mult") * hidden],
            "mtp.2.hc_head_base": [config.get("hc_mult")],
            "mtp.2.hc_head_scale": [1],
        }
        for name, expected in expected_shapes.items():
            file_name = weight_map.get(name)
            meta = all_headers.get(file_name or "", {}).get(name)
            if meta:
                require(meta.get("shape") == expected,
                        problems,
                        f"{name}: shape {meta.get('shape')} != expected {expected}")
    else:
        print("\nSafetensors headers: not checked; dspark-mtp shards are not present")

    if problems:
        print("\nProblems")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nResult: DSpark draft artifact layout looks compatible for a DS4 loader spike.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--index", default="model.safetensors.index.json")
    args = parser.parse_args()
    return inspect_artifact(args.artifact_dir, args.index)


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable


def as_pattern_entry(entry: dict):
    # minimal shim: entries are already dicts with tvm_op/ncnn_op/attrs
    class _E:
        def __init__(self, d):
            self.tvm_op = d.get("tvm_op", "")
            self.ncnn_op = d.get("ncnn_op", "")
            self.attrs = d.get("attrs", {})
            self.hardware = d.get("hardware", "")
            self.arch = d.get("arch", "")

    return _E(entry)


def has_nonzero_padding(entry: dict) -> bool:
    attrs = entry.get("attrs", {})
    pads = attrs.get("padding") or attrs.get("pad_width") or []
    return any(int(x) != 0 for x in pads)


def make_pad_entry(entry: dict) -> dict:
    attrs = entry.get("attrs", {})
    pad = attrs.get("padding") or [0, 0, 0, 0]
    return {
        "tvm_op": "relax.nn.pad",
        "ncnn_op": "Padding",
        "attrs": {
            "pad_top": int(pad[0]),
            "pad_left": int(pad[1]),
            "pad_bottom": int(pad[2]),
            "pad_right": int(pad[3]),
            "pad_value": 0,
        },
        "hardware": entry.get("hardware", ""),
        "arch": entry.get("arch", ""),
    }


def emit_metadata(output_dir: Path, matched_lines: Iterable[str], unmatched_lines: Iterable[str]) -> None:
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "op_map.txt").write_text("\n".join(matched_lines) + "\n", encoding="utf-8")
    (meta_dir / "op_unmatched.txt").write_text("\n".join(unmatched_lines) + "\n", encoding="utf-8")


def unmatched_reason(entry: dict, ncnn_name: str) -> str:
    if entry.get("tvm_op") == "relax.nn.pad":
        return "padding"
    return f"no_symbol:{ncnn_name}"

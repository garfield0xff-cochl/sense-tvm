# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

from tvm.cochl.framework.ncnn.codegen.sources import NCNN_TO_STANDALONE


def emit_metadata(output_dir: Path, matched: list[str], unmatched: list[str]) -> None:
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "op_map.txt").write_text(
        "\n".join(matched) + ("\n" if matched else ""), encoding="utf-8"
    )
    (meta_dir / "op_unmatched.txt").write_text(
        "\n".join(unmatched) + ("\n" if unmatched else ""), encoding="utf-8"
    )


def unmatched_reason(entry: dict, ncnn_name: str) -> str:
    if ncnn_name in NCNN_TO_STANDALONE:
        return "standalone mapping exists but not selected"
    if ncnn_name.startswith("permute_order_"):
        return "permute order mapped to permute_nd"
    if ncnn_name == "binary_op_broadcast":
        return "unsupported broadcast pattern (only broadcast_b/no_broadcast supported)"
    if ncnn_name.startswith("reduction_mean_"):
        return "reduction mean kernel not implemented"
    if ncnn_name in {"Convolution", "Padding", "Concat", "MatMul"}:
        return "generic op not lowered to a supported specialized kernel"
    if "conv3x3s2_neon" in ncnn_name:
        return "pack4 not eligible or non-pack4 conv3x3s2 kernel not implemented"
    return "no mapping for inferred ncnn operator"


def has_nonzero_padding(entry: dict) -> bool:
    padding = entry.get("attrs", {}).get("padding")
    if not padding:
        return False
    try:
        return any(int(v) != 0 for v in padding)
    except Exception:
        return False


def make_pad_entry(entry: dict) -> dict:
    padding = entry.get("attrs", {}).get("padding") or [0, 0, 0, 0]
    pad_top = int(padding[0]) if len(padding) > 0 else 0
    pad_left = int(padding[1]) if len(padding) > 1 else 0
    pad_bottom = int(padding[2]) if len(padding) > 2 else 0
    pad_right = int(padding[3]) if len(padding) > 3 else 0
    return {
        "tvm_op": "relax.nn.pad",
        "ncnn_op": "Padding",
        "attrs": {
            "pad_top": pad_top,
            "pad_left": pad_left,
            "pad_bottom": pad_bottom,
            "pad_right": pad_right,
            "pad_value": 0,
        },
        "hardware": entry.get("hardware", ""),
        "arch": entry.get("arch", ""),
    }


def as_pattern_entry(entry: Any):
    if hasattr(entry, "tvm_op"):
        return entry
    return type("Pattern", (), entry)()

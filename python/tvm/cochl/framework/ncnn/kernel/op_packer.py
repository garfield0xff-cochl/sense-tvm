"""ncnn op packing helpers (build_pattern_entries wrapper)."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np
from tvm import relax

from tvm.cochl.framework.ncnn.codegen.ncnn_path import NCNN_TO_STANDALONE
from tvm.cochl.framework.ncnn.relax.op.pattern_table import (
    PatternEntry,
    PatternSpec,
    build_entry_from_call,
    build_pattern_table,
    infer_ncnn_function_name,
    ncnn_patterns,
)


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


def entry_to_pattern_entry(entry: dict):
    class _E:
        def __init__(self, d):
            self.tvm_op = d.get("tvm_op", "")
            self.ncnn_op = d.get("ncnn_op", "")
            self.attrs = d.get("attrs", {})
            self.hardware = d.get("hardware", "")
            self.arch = d.get("arch", "")

    return _E(entry)


def _binary_symbol(entry: dict, ncnn_func: str) -> str | None:
    tvm_op = entry.get("tvm_op", "").split(".")[-1]
    if ncnn_func == "binary_op_vector_broadcast_b":
        if tvm_op == "maximum":
            return "binary_op_vector_broadcast_b_max_standalone"
        if tvm_op == "minimum":
            return "binary_op_vector_broadcast_b_min_standalone"
        return "binary_op_vector_broadcast_b_add_standalone"
    if ncnn_func == "binary_op_vector_no_broadcast":
        if tvm_op == "maximum":
            return "binary_op_vector_no_broadcast_max_standalone"
        if tvm_op == "minimum":
            return "binary_op_vector_no_broadcast_min_standalone"
        return "binary_op_vector_no_broadcast_add_standalone"
    return None


def resolve_symbol_name(entry: dict) -> str | None:
    ncnn_func = infer_ncnn_function_name(entry_to_pattern_entry(entry))
    if ncnn_func.startswith("permute_order_"):
        return "permute_nd"
    if ncnn_func.startswith("binary_op_"):
        sym = _binary_symbol(entry, ncnn_func)
        if sym:
            return sym
    return NCNN_TO_STANDALONE.get(ncnn_func)


def resolve_call_extern(entry: dict) -> tuple[str, str] | None:
    ncnn_func = infer_ncnn_function_name(entry_to_pattern_entry(entry))
    if ncnn_func == "conv3x3s1_pack1to4_neon":
        return "conv3x3s1_pack1to4_standalone", "conv"
    if ncnn_func == "conv3x3s2_pack1to4_neon":
        return "conv3x3s2_pack1to4_standalone", "conv"
    if ncnn_func == "conv3x3s2_neon":
        return "conv3x3s2_pack1_standalone", "conv"
    if ncnn_func == "conv1x1s1_neon":
        return "conv1x1s1_standalone", "conv"
    if ncnn_func == "convdw3x3s1_pack4_neon":
        return "convdw3x3s1_standalone", "conv"
    if ncnn_func == "convdw3x3s2_pack4_neon":
        return "convdw3x3s2_standalone", "conv"
    if ncnn_func == "binary_op_vector_broadcast_b":
        sym = _binary_symbol(entry, ncnn_func) or "binary_op_vector_broadcast_b_add_standalone"
        return sym, "binary"
    if ncnn_func == "binary_op_vector_no_broadcast":
        sym = _binary_symbol(entry, ncnn_func) or "binary_op_vector_no_broadcast_add_standalone"
        return sym, "binary"
    if ncnn_func == "binary_op_broadcast":
        return "binary_op_broadcast_add_standalone", "binary_broadcast"
    if ncnn_func == "sigmoid":
        return "sigmoid_neon_standalone", "sigmoid"
    if ncnn_func == "matmul_ta0_tb0":
        return "matmul_gemm_neon_standalone", "matmul"
    if ncnn_func == "Squeeze":
        return "squeeze_nd", "squeeze"
    if ncnn_func == "Permute" or ncnn_func.startswith("permute_order_"):
        return "permute_nd", "permute"
    if ncnn_func == "Padding":
        return "pad2d_nchw", "pad"
    if ncnn_func == "reduction_mean_2x3_k1":
        return "reduction_mean_hw_keepdims", "reduction_mean"
    return None


def build_pattern_entries(
    ir_mod: Any,
    hardware: str,
    *,
    insert_pad: bool = True,
    const_idx_map: dict[str, int] | None = None,
) -> list[dict]:
    if const_idx_map is None:
        raw_entries = build_pattern_table(ir_mod, hardware)
        pattern_entries: list[dict] = []
        for entry in raw_entries:
            if insert_pad and entry.get("tvm_op") == "relax.nn.conv2d" and has_nonzero_padding(entry):
                pattern_entries.append(make_pad_entry(entry))
            pattern_entries.append(entry)
        return pattern_entries

    # Build entries with constant index hints for stronger alignment.
    entries: list[dict] = []
    const_cache: dict[int, str] = {}

    def _hash_const_array(arr: np.ndarray) -> str:
        h = hashlib.sha1()
        h.update(arr.tobytes())
        h.update(str(arr.dtype).encode("utf-8"))
        return h.hexdigest()

    def _const_idx_for_arg(arg) -> int | None:
        if not isinstance(arg, relax.Constant):
            return None
        key = id(arg)
        if key in const_cache:
            h = const_cache[key]
        else:
            arr = arg.data.numpy()
            h = _hash_const_array(arr)
            const_cache[key] = h
        return const_idx_map.get(h)

    def _visit(expr):
        if isinstance(expr, relax.Call):
            result = build_entry_from_call(expr)
            if result is None:
                return
            op_name, ncnn_op, entry_attrs = result
            entry = PatternEntry(
                tvm_op=op_name,
                ncnn_op=ncnn_op,
                attrs=entry_attrs,
                hardware="",
                arch="",
            )
            d = asdict(entry)
            d["const_args"] = [_const_idx_for_arg(arg) for arg in expr.args]
            if insert_pad and d.get("tvm_op") == "relax.nn.conv2d" and has_nonzero_padding(d):
                entries.append(make_pad_entry(d))
            entries.append(d)

    if "main" in ir_mod.functions:
        main_func = ir_mod["main"]
        relax.analysis.post_order_visit(main_func, _visit)
    else:
        for func in ir_mod.functions.values():
            if isinstance(func, relax.Function):
                relax.analysis.post_order_visit(func, _visit)

    return entries


__all__ = [
    "PatternEntry",
    "PatternSpec",
    "entry_to_pattern_entry",
    "build_entry_from_call",
    "resolve_call_extern",
    "build_pattern_entries",
    "build_pattern_table",
    "infer_ncnn_function_name",
    "resolve_symbol_name",
    "ncnn_patterns",
]

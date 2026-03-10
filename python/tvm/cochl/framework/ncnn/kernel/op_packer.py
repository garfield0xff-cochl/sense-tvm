"""ncnn op packing helpers (build_pattern_entries wrapper)."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np
from tvm import relax

from tvm.cochl.framework.ncnn.codegen.helpers import has_nonzero_padding, make_pad_entry
from tvm.cochl.framework.ncnn.relax.op.pattern_table import (
    PatternEntry,
    PatternSpec,
    build_entry_from_call,
    build_pattern_table,
    infer_ncnn_function_name,
    ncnn_patterns,
)

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
    "build_entry_from_call",
    "build_pattern_entries",
    "build_pattern_table",
    "infer_ncnn_function_name",
    "ncnn_patterns",
]

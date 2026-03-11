# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, Iterable, List, Tuple

import tvm
from tvm.cochl.framework.relax.standalone_packer import match_relax_const_idx
from tvm.cochl.framework.ncnn.kernel.op_packer import (
    entry_to_pattern_entry,
    infer_ncnn_function_name,
    resolve_call_extern,
    resolve_symbol_name,
)
from . import sources


def _normalize_func(name: str) -> str:
    return re.sub(r"\d+$", "", name)


def _indent_block(text: str, indent: str = "    ") -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [f"{indent}{line}" for line in lines]


def _render_entry_source(
    entry_path: Path,
    model_name: str,
    proto_block: str,
    storage_block: str,
    input_shape,
    output_shape,
    num_buffers: int,
    total_storage_mb: float,
    op_count: int,
    calls: str,
    output_src: str,
    output_elems_calc: int,
    dump_ir_tensor_data: bool = False,
    trace_operator_delay: bool = False,
) -> None:
    code = sources.generate_header(proto_block)
    code += sources.generate_storage_section(storage_block)
    code += sources.generate_runtime_helpers()
    code += sources.generate_debug_helpers(
        dump_ir_tensor_data=dump_ir_tensor_data,
        trace_operator_delay=trace_operator_delay,
    )
    code += sources.generate_file_size_helper()
    code += sources.generate_inference_function(
        model_name=model_name,
        input_shape=input_shape,
        output_shape=output_shape,
        num_buffers=num_buffers,
        total_storage_mb=total_storage_mb,
        op_count=op_count,
        calls=calls,
        output_src=output_src,
        output_elems_calc=output_elems_calc,
    )
    code += sources.generate_main_function(model_name, output_elems_calc)
    entry_path.write_text(code, encoding="utf-8")


def _expected_base(entry: dict) -> str | None:
    tvm_op = entry.get("tvm_op", "")
    tvm_op = tvm_op.split(".")[-1]
    mapping = {
        "conv2d": "conv2d",
        "add": "add",
        "multiply": "multiply",
        "maximum": "maximum",
        "minimum": "minimum",
        "pad": "pad",
        "permute_dims": "transpose",
        "mean": "mean",
        "squeeze": "squeeze",
        "matmul": "matmul",
        "sigmoid": "sigmoid",
    }
    return mapping.get(tvm_op)


def _shape_of_var(plan, var: str) -> Tuple[int, ...] | None:
    if var in plan.tensors:
        return plan.tensors[var].shape
    return None


def _match_entry_plan(entry: dict, plan, plan_op) -> bool:
    base = _expected_base(entry)
    plan_base = _normalize_func(plan_op.func_name)
    # allow special-case aliases before strict match
    if base == "squeeze" and plan_base == "reshape":
        return True
    if base == "sigmoid" and plan_base in {"sigmoid", "tir_sigmoid"}:
        return True
    if base != plan_base:
        return False

    # constant index match (strong discriminator when available)
    const_args = entry.get("const_args")
    if const_args:
        plan_consts = [match_relax_const_idx(v) for v in plan_op.input_vars]
        for i, const_idx in enumerate(const_args):
            if const_idx is None:
                continue
            if i >= len(plan_consts) or plan_consts[i] != const_idx:
                return False

    # shape-aware matching for key ops
    if base == "conv2d":
        in_shape = _shape_of_var(plan, plan_op.input_vars[0])
        out_shape = _shape_of_var(plan, plan_op.output_var)
        if in_shape and entry.get("attrs", {}).get("in_channels") is not None:
            if int(in_shape[1]) != int(entry["attrs"]["in_channels"]):
                return False
        if out_shape and entry.get("attrs", {}).get("out_channels") is not None:
            if int(out_shape[1]) != int(entry["attrs"]["out_channels"]):
                return False
    if base in {"add", "multiply", "maximum", "minimum"}:
        lhs = entry.get("attrs", {}).get("lhs_shape")
        rhs = entry.get("attrs", {}).get("rhs_shape")
        in0 = _shape_of_var(plan, plan_op.input_vars[0])
        in1 = _shape_of_var(plan, plan_op.input_vars[1]) if len(plan_op.input_vars) > 1 else None
        if lhs and in0 and tuple(int(v) for v in lhs) != tuple(int(v) for v in in0):
            return False
        if rhs and in1 and tuple(int(v) for v in rhs) != tuple(int(v) for v in in1):
            return False
    if base == "pad":
        return plan_base == "pad"
    if base == "transpose":
        axes = entry.get("attrs", {}).get("axes")
        if axes is None:
            return True
        # plan does not expose axes; skip check
        return True
    if base == "mean":
        return True
    if base == "squeeze":
        return True
    if base == "matmul":
        return True
    if base == "sigmoid":
        return True
    return True


def _align_ops(ncnn_entries: List[dict], plan) -> Tuple[List[Tuple[dict, object]], List[str]]:
    aligned: List[Tuple[dict, object]] = []
    skipped: List[str] = []
    i = 0
    plan_ops = plan.operations
    for entry in ncnn_entries:
        base = _expected_base(entry)
        if base is None:
            continue
        if base == "pad":
            if i < len(plan_ops) and _match_entry_plan(entry, plan, plan_ops[i]):
                aligned.append((entry, plan_ops[i]))
                i += 1
            else:
                ncnn_name = infer_ncnn_function_name(entry_to_pattern_entry(entry))
                skipped.append(f"{entry.get('tvm_op')}:{ncnn_name}")
            continue
        while i < len(plan_ops):
            if _match_entry_plan(entry, plan, plan_ops[i]):
                aligned.append((entry, plan_ops[i]))
                i += 1
                break
            i += 1
        else:
            ncnn_name = infer_ncnn_function_name(entry_to_pattern_entry(entry))
            skipped.append(f"{entry.get('tvm_op')}:{ncnn_name}")
    return aligned, skipped


def align_entries_with_plan(ncnn_entries: List[dict], plan) -> Tuple[List[Tuple[dict, object]], List[str]]:
    return _align_ops(ncnn_entries, plan)


def _align_map(ncnn_entries: List[dict], plan) -> Tuple[Dict[int, dict], List[str]]:
    aligned, skipped = _align_ops(ncnn_entries, plan)
    op_index = {id(op): idx for idx, op in enumerate(plan.operations)}
    mapping: Dict[int, dict] = {}
    for entry, op in aligned:
        idx = op_index.get(id(op))
        if idx is None:
            continue
        mapping[idx] = entry
    return mapping, skipped


def _build_const_mapping(plan_ops) -> Dict[int, int]:
    const_indices = set()
    for op in plan_ops:
        for arg in op.input_vars:
            if 'metadata["relax.expr.Constant"]' in arg:
                idx = match_relax_const_idx(arg)
                if idx is not None:
                    const_indices.add(idx)
    sorted_idx = sorted(const_indices)
    return {const_idx: slot for slot, const_idx in enumerate(sorted_idx)}


def _load_weight_map(weights_json: Path) -> Dict[int, dict]:
    data = json.loads(weights_json.read_text(encoding="utf-8"))
    weight_map = {}
    for k, v in data.get("weights", {}).items():
        weight_map[int(k)] = v
    return weight_map


def emit_main_entry_from_plan(
    entry_path: Path,
    plan,
    ncnn_entries: List[dict],
    weights_json: Path,
    align_log: Path | None = None,
    dump_ir_tensor_data: bool = False,
    trace_operator_delay: bool = False,
) -> None:
    match_map, skipped = _align_map(ncnn_entries, plan)
    if align_log is not None:
        align_log.parent.mkdir(parents=True, exist_ok=True)
        if skipped:
            align_log.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        else:
            align_log.write_text("", encoding="utf-8")
    const_idx_to_slot = _build_const_mapping(plan.operations)
    weight_map = _load_weight_map(weights_json)

    # Build reshape alias map
    alias = {}
    for op in plan.operations:
        if op.func_name == "reshape" and len(op.input_vars) >= 1:
            alias[op.output_var] = op.input_vars[0]

    # Infer input variable name from plan ops
    known_vars = set(plan.tensors.keys()) | set(alias.keys())
    input_candidates = []
    for op in plan.operations:
        for v in op.input_vars:
            if v in known_vars:
                continue
            if 'metadata["relax.expr.Constant"]' in v:
                continue
            if "R.const(" in v:
                continue
            if v not in input_candidates:
                input_candidates.append(v)
    input_var = input_candidates[0] if len(input_candidates) == 1 else None

    extra_shapes: Dict[str, Tuple[int, ...]] = {}
    if input_var:
        for op in plan.operations:
            if op.func_name == "concatenate" and input_var in op.input_vars:
                out_shape = plan.tensors.get(op.output_var)
                if out_shape:
                    out_shape = out_shape.shape
                if out_shape and len(op.input_vars) > 0 and out_shape[-1] % len(op.input_vars) == 0:
                    # assume concat along last axis with equal channel splits
                    inferred = list(out_shape)
                    inferred[-1] = inferred[-1] // len(op.input_vars)
                    extra_shapes[input_var] = tuple(inferred)
                break

    def resolve_var(var: str) -> str:
        # resolve reshape alias
        while var in alias:
            var = alias[var]
        if input_var and var == input_var:
            return "input"
        if var in plan.tensors:
            t = plan.tensors[var]
            if t.storage_id >= 0:
                offset = t.offset_bytes // 4
                return f"&g_storage_{t.storage_id}[{offset}]"
        if 'metadata["relax.expr.Constant"]' in var:
            idx = match_relax_const_idx(var)
            if idx is not None:
                slot = const_idx_to_slot.get(idx)
                if slot is not None and slot in weight_map:
                    off = weight_map[slot]["offset"] // 4
                    return f"(float*)g_weights + {off}"
        if "R.const(0.0" in var:
            return "&scalar_zero"
        if "R.const(6.0" in var:
            return "&scalar_six"
        return "NULL"

    def tensor_shape(var: str) -> Tuple[int, ...] | None:
        while var in alias:
            var = alias[var]
        t = plan.tensors.get(var)
        if t is None:
            return extra_shapes.get(var)
        return t.shape

    def const_shape(var: str) -> Tuple[int, ...] | None:
        while var in alias:
            var = alias[var]
        if 'metadata["relax.expr.Constant"]' not in var:
            return None
        idx = match_relax_const_idx(var)
        if idx is None:
            return None
        slot = const_idx_to_slot.get(idx)
        if slot is None:
            return None
        shape = weight_map.get(slot, {}).get("shape")
        if not shape:
            return None
        return tuple(int(v) for v in shape)

    # Storage buffers
    max_storage_id = max(s.storage_id for s in plan.storages.values()) if plan.storages else -1
    storage_sizes = [0] * (max_storage_id + 1)
    for storage in plan.storages.values():
        storage_sizes[storage.storage_id] = max(storage_sizes[storage.storage_id], storage.size_bytes)

    max_elems = 0
    for t in plan.tensors.values():
        if t.shape:
            elems = 1
            for d in t.shape:
                elems *= int(d)
            if elems > max_elems:
                max_elems = elems

    storage_decls = []
    for sid, size_bytes in enumerate(storage_sizes):
        if size_bytes <= 0:
            continue
        count = size_bytes // 4
        storage_decls.append(
            f"static float __attribute__((aligned(64))) g_storage_{sid}[{count}];"
        )
    if max_elems > 0:
        storage_decls.append(
            f"static float __attribute__((aligned(64))) g_pack_tmp[{max_elems}];"
        )
        storage_decls.append(
            f"static float __attribute__((aligned(64))) g_pack_tmp2[{max_elems}];"
        )

    call_lines = []
    proto_lines = []
    op_lines = []
    op_io = []
    op_in_pack = []
    op_out_pack = []
    op_in_shape = []
    op_out_shape = []

    # track pack state per tensor var (1 or 4)
    pack_state: Dict[str, int] = {}

    def _pack_for_var(var: str) -> int:
        while var in alias:
            var = alias[var]
        return pack_state.get(var, 1)

    def _shape_for_var(var: str) -> Tuple[int, ...] | None:
        return tensor_shape(var)

    def _elems_from_shape(shape: Tuple[int, ...] | None) -> int:
        if not shape:
            return 0
        size = 1
        for d in shape:
            size *= int(d)
        return size

    def _append_op_io(
        in_ptr: str,
        out_ptr: str,
        in_elems: int,
        out_elems: int,
        in_pack: int | None = None,
        out_pack: int | None = None,
        in_shape: Tuple[int, ...] | None = None,
        out_shape: Tuple[int, ...] | None = None,
    ) -> None:
        op_io.append((in_ptr, out_ptr, in_elems, out_elems))
        op_in_pack.append(in_pack if in_pack is not None else 1)
        op_out_pack.append(out_pack if out_pack is not None else 1)
        op_in_shape.append(in_shape if in_shape is not None else (1, 1, 1, 1))
        op_out_shape.append(out_shape if out_shape is not None else (1, 1, 1, 1))

    def _kernel_tm_dims(inch: int, outch: int, kernel_w: int, kernel_h: int) -> Tuple[int, int, int]:
        maxk = kernel_w * kernel_h
        # rpi2 target: __ARM_NEON enabled, __aarch64__ disabled
        if outch >= 4:
            if inch >= 4:
                return (
                    4 * 4 * maxk,
                    inch // 4 + (inch % 4) // 2 + inch % 2,
                    outch // 4 + (outch % 4) // 2 + outch % 2,
                )
            if inch >= 2:
                return (
                    4 * 2 * maxk,
                    inch // 2 + inch % 2,
                    outch // 4 + (outch % 4) // 2 + outch % 2,
                )
            return (
                4 * maxk,
                inch,
                outch // 4 + (outch % 4) // 2 + outch % 2,
            )
        if outch >= 2:
            if inch >= 4:
                return (
                    2 * 4 * maxk,
                    inch // 4 + (inch % 4) // 2 + inch % 2,
                    outch // 2 + outch % 2,
                )
            if inch >= 2:
                return (
                    2 * 2 * maxk,
                    inch // 2 + inch % 2,
                    outch // 2 + outch % 2,
                )
            return (
                2 * maxk,
                inch,
                outch // 2 + outch % 2,
            )
        if inch >= 4:
            return (
                4 * maxk,
                inch // 4 + (inch % 4) // 2 + inch % 2,
                outch,
            )
        if inch >= 2:
            return (
                2 * maxk,
                inch // 2 + inch % 2,
                outch,
            )
        return (maxk, inch, outch)

    for op_idx, op in enumerate(plan.operations):
        entry = match_map.get(op_idx)
        inputs = [resolve_var(v) for v in op.input_vars]
        while len(inputs) < 2:
            inputs.append("NULL")
        output = resolve_var(op.output_var)
        in_shape0 = tensor_shape(op.input_vars[0]) if op.input_vars else None
        out_shape = tensor_shape(op.output_var)
        in_elems = _elems_from_shape(in_shape0) if in_shape0 else 0
        out_elems = _elems_from_shape(out_shape) if out_shape else 0

        if entry is not None:
            ncnn_name = infer_ncnn_function_name(entry_to_pattern_entry(entry))
            if entry.get("tvm_op") == "relax.multiply" and len(op.input_vars) == 1:
                scalar = 1.0
                scalar_map = getattr(plan, "scalar_values", {})
                if isinstance(scalar_map, dict) and scalar_map.get("multiply") is not None:
                    scalar = float(scalar_map["multiply"])
                op_label = f"{entry['tvm_op']}:{ncnn_name}:scalar"
                op_lines.append(("fallback_mul", "fallback_mul", op_label))
                _append_op_io(
                    inputs[0],
                    output,
                    in_elems,
                    out_elems,
                    in_pack=_pack_for_var(op.input_vars[0]) if op.input_vars else 1,
                    out_pack=1,
                    in_shape=in_shape0,
                    out_shape=out_shape,
                )
                proto_lines.append("static void mul_scalar(const float* in, float* out, long size, float s);")
                size = out_elems or in_elems or 1
                call_lines.append(f"    mul_scalar({inputs[0]}, {output}, {size}, {scalar:.8f}f);")
                pack_state[op.output_var] = 1
                continue

            symbol_info = resolve_call_extern(entry)
            if symbol_info is None:
                continue
            symbol, kind = symbol_info
            op_label = f"{entry['tvm_op']}:{ncnn_name}"
            op_lines.append((symbol, kind, op_label))
            _append_op_io(
                inputs[0],
                output,
                in_elems,
                out_elems,
                in_pack=_pack_for_var(op.input_vars[0]) if op.input_vars else 1,
                out_pack=1,
                in_shape=in_shape0,
                out_shape=out_shape,
            )

            if kind == "conv":
                # pack handling for conv
                desired_in_pack = 1
                desired_out_pack = 1
                if ncnn_name in {"conv3x3s1_pack1to4_neon", "conv3x3s2_pack1to4_neon"}:
                    desired_in_pack = 1
                    desired_out_pack = 4
                elif ncnn_name == "convdw3x3s1_pack4_neon":
                    desired_in_pack = 4
                    desired_out_pack = 4
                elif ncnn_name == "convdw3x3s2_pack4_neon":
                    desired_in_pack = 4
                    desired_out_pack = 4

                pre_lines: List[str] = []
                in_ptr = inputs[0]
                in_pack = _pack_for_var(op.input_vars[0])

                in_shape = in_shape0 or (1, 1, 1, 1)
                out_shape_use = out_shape or (1, 1, 1, 1)
                in_c = int(in_shape[1]) if len(in_shape) > 1 else 1
                in_h = int(in_shape[2]) if len(in_shape) > 2 else 1
                in_w = int(in_shape[3]) if len(in_shape) > 3 else 1
                out_c = int(out_shape_use[1]) if len(out_shape_use) > 1 else 1
                out_h = int(out_shape_use[2]) if len(out_shape_use) > 2 else 1
                out_w = int(out_shape_use[3]) if len(out_shape_use) > 3 else 1
                weight_ptr = inputs[1] if len(inputs) > 1 else "NULL"

                # apply padding for conv kernels that expect padded input
                pad = entry.get("attrs", {}).get("padding", [0, 0, 0, 0])
                if pad and any(int(x) != 0 for x in pad):
                    pad_top = int(pad[0])
                    pad_left = int(pad[1])
                    pad_bottom = int(pad[2])
                    pad_right = int(pad[3])
                    n = int(in_shape[0]) if len(in_shape) > 0 else 1
                    # ensure pack1 input for padding
                    if in_pack == 4:
                        pre_lines.append(
                            f"    pack4_to_pack1({inputs[0]}, g_pack_tmp, {n}, {in_c}, {in_h}, {in_w});"
                        )
                        in_ptr = "g_pack_tmp"
                        in_pack = 1
                    proto_lines.append(
                        "int pad2d_nchw(const float*, float*, int, int, int, int, int, int, int, int, float);"
                    )
                    pre_lines.append(
                        f"    pad2d_nchw({in_ptr}, g_pack_tmp2, {n}, {in_c}, {in_h}, {in_w}, {pad_top}, {pad_left}, {pad_bottom}, {pad_right}, 0.0f);"
                    )
                    in_ptr = "g_pack_tmp2"
                    in_h = in_h + pad_top + pad_bottom
                    in_w = in_w + pad_left + pad_right
                    in_shape = (n, in_c, in_h, in_w)

                if in_pack != desired_in_pack:
                    n = int(in_shape[0]) if len(in_shape) > 0 else 1
                    c = int(in_shape[1]) if len(in_shape) > 1 else 1
                    h = int(in_shape[2]) if len(in_shape) > 2 else 1
                    w = int(in_shape[3]) if len(in_shape) > 3 else 1
                    if desired_in_pack == 4:
                        pre_lines.append(f"    pack1_to_pack4({in_ptr}, g_pack_tmp, {n}, {c}, {h}, {w});")
                    else:
                        pre_lines.append(f"    pack4_to_pack1({in_ptr}, g_pack_tmp, {n}, {c}, {h}, {w});")
                    in_ptr = "g_pack_tmp"
                # update debug pack metadata for dumps (keep original input pointer)
                if op_in_pack:
                    op_in_pack[-1] = in_pack
                    op_out_pack[-1] = desired_out_pack
                    op_in_shape[-1] = in_shape
                    op_out_shape[-1] = out_shape_use
                k_h = int(entry.get("attrs", {}).get("kernel_h", 0))
                k_w = int(entry.get("attrs", {}).get("kernel_w", 0))
                stride = int(entry.get("attrs", {}).get("strides", [1, 1])[0])
                dilation = int(entry.get("attrs", {}).get("dilation", [1, 1])[0])
                groups = int(entry.get("attrs", {}).get("groups", 1))

                proto_lines.append(
                    f"int {symbol}(const float*, const float*, const float*, float*, int, int, int, int, int, int);"
                )
                call_lines.append(
                    "\n".join(pre_lines + [f"    {symbol}({in_ptr}, {weight_ptr}, NULL, {output}, {in_c}, {in_h}, {in_w}, {out_c}, {out_h}, {out_w});"])
                )
                pack_state[op.output_var] = desired_out_pack
            elif kind == "binary":
                size = in_elems or 1
                pre_lines = []
                in_ptr0 = inputs[0]
                in_ptr1 = inputs[1]
                broadcast_kind = entry.get("attrs", {}).get("broadcast")
                # channel-wise bias add: rhs shape [1,C,1,1]
                rhs_shape = entry.get("attrs", {}).get("rhs_shape")
                lhs_shape = entry.get("attrs", {}).get("lhs_shape")
                if rhs_shape and lhs_shape and len(rhs_shape) == 4 and rhs_shape[2] == 1 and rhs_shape[3] == 1:
                    n = int(lhs_shape[0])
                    c = int(lhs_shape[1])
                    h = int(lhs_shape[2])
                    w = int(lhs_shape[3])
                    in_pack0 = _pack_for_var(op.input_vars[0])
                    if in_pack0 == 4:
                        proto_lines.append(
                            "static void add_channel_bias_pack4(const float* a, const float* b, float* out, int n, int c, int h, int w);"
                        )
                        call_lines.append(
                            f"    add_channel_bias_pack4({inputs[0]}, {inputs[1]}, {output}, {n}, {c}, {h}, {w});"
                        )
                        if op_out_pack:
                            op_out_pack[-1] = 4
                        pack_state[op.output_var] = 4
                    else:
                        proto_lines.append(
                            "static void add_channel_bias(const float* a, const float* b, float* out, int n, int c, int inner);"
                        )
                        call_lines.append(
                            f"    add_channel_bias({inputs[0]}, {inputs[1]}, {output}, {n}, {c}, {h * w});"
                        )
                        pack_state[op.output_var] = 1
                    continue
                in_pack0 = _pack_for_var(op.input_vars[0]) if op.input_vars else 1
                in_pack1 = _pack_for_var(op.input_vars[1]) if len(op.input_vars) > 1 and op.input_vars[1] in plan.tensors else 1
                rhs_is_scalar = (broadcast_kind == "broadcast_b_scalar")
                can_use_pack4 = (in_pack0 == 4) and (in_pack1 == 4 or rhs_is_scalar)
                if not can_use_pack4 and in_pack0 == 4:
                    in_shape = _shape_for_var(op.input_vars[0]) or (1, 1, 1, 1)
                    n = int(in_shape[0]) if len(in_shape) > 0 else 1
                    c = int(in_shape[1]) if len(in_shape) > 1 else 1
                    h = int(in_shape[2]) if len(in_shape) > 2 else 1
                    w = int(in_shape[3]) if len(in_shape) > 3 else 1
                    pre_lines.append(f"    pack4_to_pack1({inputs[0]}, g_pack_tmp, {n}, {c}, {h}, {w});")
                    in_ptr0 = "g_pack_tmp"
                if not can_use_pack4 and len(op.input_vars) > 1 and op.input_vars[1] in plan.tensors:
                    if in_pack1 == 4:
                        in_shape = _shape_for_var(op.input_vars[1]) or (1, 1, 1, 1)
                        n = int(in_shape[0]) if len(in_shape) > 0 else 1
                        c = int(in_shape[1]) if len(in_shape) > 1 else 1
                        h = int(in_shape[2]) if len(in_shape) > 2 else 1
                        w = int(in_shape[3]) if len(in_shape) > 3 else 1
                        pre_lines.append(f"    pack4_to_pack1({inputs[1]}, g_pack_tmp2, {n}, {c}, {h}, {w});")
                        in_ptr1 = "g_pack_tmp2"
                if symbol == "binary_op_vector_no_broadcast_add_standalone":
                    proto_lines.append(
                        f"int {symbol}(const float*, const float*, float*, int);"
                    )
                    call_lines.append("\n".join(pre_lines + [f"    {symbol}({in_ptr0}, {in_ptr1}, {output}, {size});"]))
                else:
                    elempack = 4 if (can_use_pack4 and not rhs_is_scalar) else 1
                    proto_lines.append(
                        f"int {symbol}(const float*, const float*, float*, int, int);"
                    )
                    call_lines.append("\n".join(pre_lines + [f"    {symbol}({in_ptr0}, {in_ptr1}, {output}, {size}, {elempack});"]))
                pack_state[op.output_var] = 4 if can_use_pack4 else 1
            elif kind == "binary_broadcast":
                size = in_elems or 1
                pre_lines = []
                in_ptr0 = inputs[0]
                rhs_shape = entry.get("attrs", {}).get("rhs_shape")
                lhs_shape = entry.get("attrs", {}).get("lhs_shape")
                if rhs_shape and lhs_shape and len(rhs_shape) == 4 and rhs_shape[2] == 1 and rhs_shape[3] == 1:
                    n = int(lhs_shape[0])
                    c = int(lhs_shape[1])
                    h = int(lhs_shape[2])
                    w = int(lhs_shape[3])
                    proto_lines.append(
                        "static void add_channel_bias(const float* a, const float* b, float* out, int n, int c, int inner);"
                    )
                    call_lines.append(
                        f"    add_channel_bias({inputs[0]}, {inputs[1]}, {output}, {n}, {c}, {h * w});"
                    )
                    pack_state[op.output_var] = 1
                    continue
                if op.input_vars:
                    in_pack0 = _pack_for_var(op.input_vars[0])
                    if in_pack0 == 4:
                        in_shape = _shape_for_var(op.input_vars[0]) or (1, 1, 1, 1)
                        n = int(in_shape[0]) if len(in_shape) > 0 else 1
                        c = int(in_shape[1]) if len(in_shape) > 1 else 1
                        h = int(in_shape[2]) if len(in_shape) > 2 else 1
                        w = int(in_shape[3]) if len(in_shape) > 3 else 1
                        pre_lines.append(f"    pack4_to_pack1({inputs[0]}, g_pack_tmp, {n}, {c}, {h}, {w});")
                        in_ptr0 = "g_pack_tmp"
                proto_lines.append(
                    f"int {symbol}(const float*, const float*, float*, int, int);"
                )
                call_lines.append("\n".join(pre_lines + [f"    {symbol}({in_ptr0}, {inputs[1]}, {output}, 1, {size});"]))
                pack_state[op.output_var] = 1
            elif kind == "sigmoid":
                size = out_elems or 1
                proto_lines.append(f"int {symbol}(float*, int);")
                call_seq = []
                if inputs and inputs[0] != output:
                    proto_lines.append("static void reshape_copy(const float* in, float* out, long size);")
                    call_seq.append(f"    reshape_copy({inputs[0]}, {output}, {size});")
                call_seq.append(f"    {symbol}({output}, {size});")
                call_lines.append("\n".join(call_seq))
                pack_state[op.output_var] = _pack_for_var(op.input_vars[0]) if op.input_vars else 1
            elif kind == "matmul":
                a_shape = tensor_shape(op.input_vars[0]) or (1, 1)
                b_shape = tensor_shape(op.input_vars[1]) or const_shape(op.input_vars[1]) or (1, 1)
                m = int(a_shape[0]) if len(a_shape) > 0 else 1
                k = int(a_shape[1]) if len(a_shape) > 1 else 1
                n = int(b_shape[1]) if len(b_shape) > 1 else 1
                proto_lines.append(
                    f"int {symbol}(const float*, const float*, const float*, float*, int, int, int);"
                )
                call_lines.append(f"    {symbol}({inputs[0]}, {inputs[1]}, NULL, {output}, {m}, {k}, {n});")
                pack_state[op.output_var] = 1
            elif kind == "squeeze":
                in_shape = in_shape0 or (1,)
                axes = entry.get("attrs", {}).get("axis") or []
                proto_lines.append(
                    f"int {symbol}(const float*, float*, const int*, int, const int*, int);"
                )
                shape_arr = ", ".join(str(int(v)) for v in in_shape)
                axes_arr = ", ".join(str(int(v)) for v in axes) if axes else "0"
                call_lines.append(
                    f"    {{ int shape[] = {{{shape_arr}}}; int axes[] = {{{axes_arr}}}; {symbol}({inputs[0]}, {output}, shape, {len(in_shape)}, axes, {len(axes) if axes else 1}); }}"
                )
                pack_state[op.output_var] = _pack_for_var(op.input_vars[0]) if op.input_vars else 1
            elif kind == "permute":
                in_shape = in_shape0 or (1,)
                axes = entry.get("attrs", {}).get("axes") or []
                proto_lines.append(
                    f"int {symbol}(const float*, float*, const int*, int, const int*);"
                )
                shape_arr = ", ".join(str(int(v)) for v in in_shape)
                axes_arr = ", ".join(str(int(v)) for v in axes) if axes else "0"
                call_lines.append(
                    f"    {{ int shape[] = {{{shape_arr}}}; int perm[] = {{{axes_arr}}}; {symbol}({inputs[0]}, {output}, shape, {len(in_shape)}, perm); }}"
                )
                pack_state[op.output_var] = _pack_for_var(op.input_vars[0]) if op.input_vars else 1
            elif kind == "pad":
                in_shape = in_shape0 or (1, 1, 1, 1)
                out_shape_use = out_shape or in_shape
                n = int(in_shape[0]) if len(in_shape) > 0 else 1
                c = int(in_shape[1]) if len(in_shape) > 1 else 1
                h = int(in_shape[2]) if len(in_shape) > 2 else 1
                w = int(in_shape[3]) if len(in_shape) > 3 else 1
                out_h = int(out_shape_use[2]) if len(out_shape_use) > 2 else h
                out_w = int(out_shape_use[3]) if len(out_shape_use) > 3 else w
                pad_top = int(entry.get("attrs", {}).get("pad_top", 0))
                pad_left = int(entry.get("attrs", {}).get("pad_left", 0))
                pad_bottom = int(entry.get("attrs", {}).get("pad_bottom", 0))
                pad_right = int(entry.get("attrs", {}).get("pad_right", 0))
                proto_lines.append(
                    f"int {symbol}(const float*, float*, int, int, int, int, int, int, int, int, float);"
                )
                in_pack = _pack_for_var(op.input_vars[0]) if op.input_vars else 1
                if in_pack == 4:
                    call_lines.append(
                        "\n".join(
                            [
                                f"    pack4_to_pack1({inputs[0]}, g_pack_tmp, {n}, {c}, {h}, {w});",
                                f"    {symbol}(g_pack_tmp, g_pack_tmp2, {n}, {c}, {h}, {w}, {pad_top}, {pad_left}, {pad_bottom}, {pad_right}, 0.0f);",
                                f"    pack1_to_pack4(g_pack_tmp2, {output}, {n}, {c}, {out_h}, {out_w});",
                            ]
                        )
                    )
                    if op_out_pack:
                        op_out_pack[-1] = 4
                    pack_state[op.output_var] = 4
                else:
                    call_lines.append(
                        f"    {symbol}({inputs[0]}, {output}, {n}, {c}, {h}, {w}, {pad_top}, {pad_left}, {pad_bottom}, {pad_right}, 0.0f);"
                    )
                    pack_state[op.output_var] = 1
            elif kind == "reduction_mean":
                in_shape = in_shape0 or (1, 1, 1, 1)
                n = int(in_shape[0]) if len(in_shape) > 0 else 1
                c = int(in_shape[1]) if len(in_shape) > 1 else 1
                h = int(in_shape[2]) if len(in_shape) > 2 else 1
                w = int(in_shape[3]) if len(in_shape) > 3 else 1
                proto_lines.append(
                    f"int {symbol}(const float*, float*, int, int, int, int);"
                )
                call_lines.append(
                    f"    {symbol}({inputs[0]}, {output}, {n}, {c}, {h}, {w});"
                )
                pack_state[op.output_var] = 1
            continue

        # Fallback ops not covered by ncnn matching
        base = _normalize_func(op.func_name)
        if base == "pad":
            op_label = "relax.nn.pad:pad2d_nchw"
            op_lines.append(("ncnn", "ncnn", op_label))
        else:
            op_label = f"fallback:{op.func_name}"
            op_lines.append(("fallback", "fallback", op_label))
        _append_op_io(
            inputs[0],
            output,
            in_elems,
            out_elems,
            in_pack=_pack_for_var(op.input_vars[0]) if op.input_vars else 1,
            out_pack=_pack_for_var(op.input_vars[0]) if op.input_vars else 1,
            in_shape=in_shape0,
            out_shape=out_shape,
        )
        if base == "concatenate":
            # assume concat on last axis with equal channel splits
            out_shape_use = out_shape or (1, 1, 1, len(op.input_vars))
            axis = len(out_shape_use) - 1
            n = int(out_shape_use[0]) if len(out_shape_use) > 0 else 1
            h = int(out_shape_use[1]) if len(out_shape_use) > 1 else 1
            w = int(out_shape_use[2]) if len(out_shape_use) > 2 else 1
            c_total = int(out_shape_use[3]) if len(out_shape_use) > 3 else len(op.input_vars)
            in_c_each = c_total // max(len(op.input_vars), 1)
            proto_lines.append("static void concat_last_axis(const float** inputs, int num_inputs, float* output, int n, int h, int w, int c_each);")
            call_lines.append(
                f"    {{ const float* ins[{len(op.input_vars)}] = {{{', '.join(inputs[:len(op.input_vars)])}}}; concat_last_axis(ins, {len(op.input_vars)}, {output}, {n}, {h}, {w}, {in_c_each}); }}"
            )
            pack_state[op.output_var] = 1
        elif base == "reshape":
            proto_lines.append("static void reshape_copy(const float* in, float* out, long size);")
            size = out_elems or in_elems or 1
            if in_elems and out_elems:
                size = min(in_elems, out_elems)
            call_lines.append(f"    reshape_copy({inputs[0]}, {output}, {size});")
            pack_state[op.output_var] = _pack_for_var(op.input_vars[0]) if op.input_vars else 1
        elif base == "pad":
            in_shape = in_shape0 or (1, 1, 1, 1)
            out_shape_use = out_shape or in_shape
            n = int(in_shape[0]) if len(in_shape) > 0 else 1
            c = int(in_shape[1]) if len(in_shape) > 1 else 1
            h = int(in_shape[2]) if len(in_shape) > 2 else 1
            w = int(in_shape[3]) if len(in_shape) > 3 else 1
            out_h = int(out_shape_use[2]) if len(out_shape_use) > 2 else h
            out_w = int(out_shape_use[3]) if len(out_shape_use) > 3 else w
            attrs = entry.get("attrs", {}) if entry is not None else {}
            pad_top = int(attrs.get("pad_top", 0))
            pad_left = int(attrs.get("pad_left", 0))
            pad_bottom = int(attrs.get("pad_bottom", 0))
            pad_right = int(attrs.get("pad_right", 0))
            if pad_top == 0 and pad_left == 0 and pad_bottom == 0 and pad_right == 0:
                pad_bottom = max(out_h - h, 0)
                pad_right = max(out_w - w, 0)
            proto_lines.append(
                "int pad2d_nchw(const float*, float*, int, int, int, int, int, int, int, int, float);"
            )
            in_pack = _pack_for_var(op.input_vars[0]) if op.input_vars else 1
            if in_pack == 4:
                call_lines.append(
                    "\n".join(
                        [
                            f"    pack4_to_pack1({inputs[0]}, g_pack_tmp, {n}, {c}, {h}, {w});",
                            f"    pad2d_nchw(g_pack_tmp, g_pack_tmp2, {n}, {c}, {h}, {w}, {pad_top}, {pad_left}, {pad_bottom}, {pad_right}, 0.0f);",
                            f"    pack1_to_pack4(g_pack_tmp2, {output}, {n}, {c}, {out_h}, {out_w});",
                        ]
                    )
                )
                if op_out_pack:
                    op_out_pack[-1] = 4
                pack_state[op.output_var] = 4
            else:
                call_lines.append(
                    f"    pad2d_nchw({inputs[0]}, {output}, {n}, {c}, {h}, {w}, {pad_top}, {pad_left}, {pad_bottom}, {pad_right}, 0.0f);"
                )
                pack_state[op.output_var] = 1
        else:
            # unsupported fallback: no-op with memcpy if possible
            proto_lines.append("static void reshape_copy(const float* in, float* out, long size);")
            size = out_elems or in_elems or 1
            if in_elems and out_elems:
                size = min(in_elems, out_elems)
            call_lines.append(f"    reshape_copy({inputs[0]}, {output}, {size});")
            pack_state[op.output_var] = _pack_for_var(op.input_vars[0]) if op.input_vars else 1

    # determine output copy source
    output_src = None
    output_elems_calc = 0
    if plan.operations:
        output_var = plan.operations[-1].output_var
        output_src = resolve_var(output_var)
        out_shape = tensor_shape(output_var)
        if out_shape:
            total = 1
            for d in out_shape:
                total *= int(d)
            output_elems_calc = total

    debug_calls = []
    for idx, ((symbol, kind, op_label), raw_call) in enumerate(zip(op_lines, call_lines)):
        in_ptr, out_ptr, in_elems_use, out_elems_use = op_io[idx]
        in_pack = op_in_pack[idx] if idx < len(op_in_pack) else 1
        out_pack = op_out_pack[idx] if idx < len(op_out_pack) else 1
        in_shape = op_in_shape[idx] if idx < len(op_in_shape) else (1, 1, 1, 1)
        out_shape = op_out_shape[idx] if idx < len(op_out_shape) else (1, 1, 1, 1)
        in_n, in_c, in_h, in_w = (int(in_shape[0]), int(in_shape[1]), int(in_shape[2]), int(in_shape[3])) if len(in_shape) >= 4 else (1, 1, 1, 1)
        out_n, out_c, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]), int(out_shape[3])) if len(out_shape) >= 4 else (1, 1, 1, 1)

        if debug_calls:
            debug_calls.append("")
        debug_calls.append(f"    // op {idx}: {op_label}")
        if dump_ir_tensor_data:
            if in_pack == 4:
                debug_calls.append(f"    pack4_to_pack1({in_ptr}, g_pack_tmp2, {in_n}, {in_c}, {in_h}, {in_w});")
                debug_calls.append(f"    dump_tensor_data(\"Op{idx + 1}: input ({op_label})\", g_pack_tmp2, {in_elems_use}, 64);")
            else:
                debug_calls.append(f"    dump_tensor_data(\"Op{idx + 1}: input ({op_label})\", {in_ptr}, {in_elems_use}, 64);")
        if trace_operator_delay:
            debug_calls.append(f"    clock_t op_start_{idx} = clock();")
        debug_calls.extend(_indent_block(raw_call))
        if trace_operator_delay:
            debug_calls.append(f"    trace_operator_delay(\"Op{idx + 1}: {op_label}\", op_start_{idx}, clock());")
        if dump_ir_tensor_data:
            if out_pack == 4:
                debug_calls.append(f"    pack4_to_pack1({out_ptr}, g_pack_tmp2, {out_n}, {out_c}, {out_h}, {out_w});")
                debug_calls.append(f"    dump_tensor_data(\"Op{idx + 1}: output ({op_label})\", g_pack_tmp2, {out_elems_use}, 64);")
            else:
                debug_calls.append(f"    dump_tensor_data(\"Op{idx + 1}: output ({op_label})\", {out_ptr}, {out_elems_use}, 64);")

    calls = "\n".join(debug_calls) if debug_calls else "    (void)a;"
    proto_block = "\n".join(sorted(set(proto_lines)))
    storage_block = "\n".join(storage_decls)
    input_shape = tensor_shape(input_var) if input_var else ()
    final_output_shape = tensor_shape(plan.operations[-1].output_var) if plan.operations else ()
    num_buffers = len(storage_sizes) + (2 if max_elems > 0 else 0)
    total_storage_mb = (
        sum(size for size in storage_sizes if size > 0) + (max_elems * 4 * 2 if max_elems > 0 else 0)
    ) / (1024 * 1024)
    _render_entry_source(
        entry_path=entry_path,
        model_name=entry_path.stem,
        proto_block=proto_block,
        storage_block=storage_block,
        input_shape=input_shape,
        output_shape=final_output_shape or (),
        num_buffers=num_buffers,
        total_storage_mb=total_storage_mb,
        op_count=len(op_lines),
        calls=calls,
        output_src=output_src if output_src else "NULL",
        output_elems_calc=output_elems_calc,
        dump_ir_tensor_data=dump_ir_tensor_data,
        trace_operator_delay=trace_operator_delay,
    )

# SPDX-License-Identifier: Apache-2.0
# Legacy entrygen (kept for compatibility)


def emit_main_entry(
    entry_path: Path,
    pattern_entries: Iterable[dict],
    dump_ir_tensor_data: bool = False,
    trace_operator_delay: bool = False,
) -> None:
    output_elems_calc = 1
    call_lines = []
    proto_lines = []
    op_lines = []
    for entry in pattern_entries:
        symbol_info = resolve_call_extern(entry)
        if symbol_info is None:
            continue
        symbol, kind = symbol_info
        ncnn_name = infer_ncnn_function_name(entry_to_pattern_entry(entry))
        op_label = f"{entry['tvm_op']}:{ncnn_name}"
        op_lines.append((symbol, kind, op_label))
        if kind == "conv":
            proto_lines.append(
                f"int {symbol}(const float*, const float*, const float*, float*, int, int, int, int, int, int);"
            )
            call_lines.append(f"    {symbol}(a, w, b, c, 1, 1, 1, 1, 1, 1);")
        elif kind == "binary":
            if symbol == "binary_op_vector_no_broadcast_add_standalone":
                proto_lines.append(
                    f"int {symbol}(const float*, const float*, float*, int);"
                )
                call_lines.append(f"    {symbol}(a, b, c, 1);")
            else:
                proto_lines.append(
                    f"int {symbol}(const float*, const float*, float*, int, int);"
                )
                call_lines.append(f"    {symbol}(a, b, c, 1, 1);")
        elif kind == "binary_broadcast":
            proto_lines.append(
                f"int {symbol}(const float*, const float*, float*, int, int);"
            )
            call_lines.append(f"    {symbol}(a, b, c, 1, 1);")
        elif kind == "sigmoid":
            proto_lines.append(f"int {symbol}(float*, int);")
            call_lines.append(f"    {symbol}(a, 1);")
        elif kind == "matmul":
            proto_lines.append(
                f"int {symbol}(const float*, const float*, const float*, float*, int, int, int);"
            )
            call_lines.append(f"    {symbol}(a, b, c, d, 1, 1, 1);")
        elif kind == "squeeze":
            proto_lines.append(
                f"int {symbol}(const float*, float*, const int*, int, const int*, int);"
            )
            call_lines.append(f"    {symbol}(a, c, shape, 1, axes, 1);")
        elif kind == "permute":
            proto_lines.append(
                f"int {symbol}(const float*, float*, const int*, int, const int*);"
            )
            call_lines.append(f"    {symbol}(a, c, shape, 1, perm);")
        elif kind == "reduction_mean":
            proto_lines.append(
                f"int {symbol}(const float*, float*, int, int, int, int);"
            )
            call_lines.append(f"    {symbol}(a, c, 1, 1, 1, 1);")
        elif kind == "pad":
            proto_lines.append(
                f"int {symbol}(const float*, float*, int, int, int, int, int, int, int, int, float);"
            )
            call_lines.append(f"    {symbol}(a, c, 1, 1, 1, 1, 0, 0, 0, 0, 0.0f);")

    debug_calls = []
    for idx, ((symbol, kind, op_label), raw_call) in enumerate(zip(op_lines, call_lines)):
        if debug_calls:
            debug_calls.append("")
        debug_calls.append(f"    // op {idx}: {op_label}")
        if dump_ir_tensor_data:
            debug_calls.append(f"    dump_tensor_data(\"Op{idx + 1}: input ({op_label})\", a, in_elems, 64);")
        if trace_operator_delay:
            debug_calls.append(f"    clock_t op_start_{idx} = clock();")
        debug_calls.extend(_indent_block(raw_call))
        if trace_operator_delay:
            debug_calls.append(f"    trace_operator_delay(\"Op{idx + 1}: {op_label}\", op_start_{idx}, clock());")
        if dump_ir_tensor_data:
            debug_calls.append(f"    dump_tensor_data(\"Op{idx + 1}: output ({op_label})\", c, out_elems_use, 64);")

    calls = "\n".join(debug_calls) if debug_calls else "    (void)a;"
    proto_block = "\n".join(sorted(set(proto_lines)))
    _render_entry_source(
        entry_path=entry_path,
        model_name=entry_path.stem,
        proto_block=proto_block,
        storage_block="",
        input_shape=(),
        output_shape=(),
        num_buffers=0,
        total_storage_mb=0.0,
        op_count=len(op_lines),
        calls=calls,
        output_src="output",
        output_elems_calc=output_elems_calc,
        dump_ir_tensor_data=dump_ir_tensor_data,
        trace_operator_delay=trace_operator_delay,
    )


# ---- ncnn codegen entry ----

def _codegen_impl(
    ir_mod,
    input_name,
    input_shape,
    output_shape,
    weights,
    weight_order,
    output_dir,
    model_name="sense_model",
    save_metadata=True,
    **kwargs,
) -> bool:
    """ncnn backend codegen entry (Sense pipeline)."""
    from tvm.cochl.framework.ncnn.kernel.op_packer import build_pattern_entries
    from tvm.cochl.framework.ncnn.kernel.weight_packer import NcnnWeightPacker
    from tvm.cochl.framework.ncnn.codegen.ncnn_path import NCNN_HEADER, NCNN_TO_STANDALONE
    from tvm.cochl.framework.ncnn.codegen.libgen import build_call_extern_module, collect_neon_sources, write_lib0_with_impl
    from tvm.cochl.framework.ncnn.codegen.memory_plan import build_plan
    import hashlib
    import json
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_ir_tensor_data = bool(kwargs.get("dump_ir_tensor_data", False))
    trace_operator_delay = bool(kwargs.get("trace_operator_delay", False))
    metadata_dir = output_dir / "metadata"
    for stale_name in ("op_delay.txt",):
        stale_path = metadata_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # Build const index map (hash -> const idx, shape-agnostic)
    const_idx_map: dict[str, int] = {}
    for idx, name in enumerate(weight_order):
        arr = weights[name]
        h = hashlib.sha1()
        h.update(arr.tobytes())
        h.update(str(arr.dtype).encode("utf-8"))
        const_idx_map[h.hexdigest()] = idx

    # Build pattern table and operator map (with const indices)
    pattern_entries = build_pattern_entries(
        ir_mod,
        "rpi2",
        insert_pad=True,
        const_idx_map=const_idx_map,
    )

    matched_symbols: set[str] = set()
    for idx, entry in enumerate(pattern_entries):
        symbol = resolve_symbol_name(entry)
        if symbol is None:
            continue
        else:
            matched_symbols.add(symbol)

    matched_symbols.update(
        {
            "conv3x3s1_pack1to4_standalone",
            "conv3x3s2_pack1to4_standalone",
            "conv3x3s2_neon_standalone",
            "conv1x1s1_standalone",
            "convdw3x3s1_standalone",
            "convdw3x3s2_standalone",
        }
    )

    if any(entry.get("tvm_op") == "relax.nn.conv2d" for entry in pattern_entries):
        matched_symbols.add("convolution_transform_kernel_packed_standalone")
        matched_symbols.add("convolution_packed_neon_standalone")

    # Build tvm_c memory plan (for const index mapping)
    model_path = kwargs.get("model_path") or kwargs.get("onnx_path")
    if model_path is None:
        raise ValueError("ncnn codegen requires model_path/onnx_path for memory plan")
    plan = build_plan(model_path)
    if any(op.func_name == "pad" for op in plan.operations):
        matched_symbols.add("pad2d_nchw")

    # Pack weights for NCNN backend with pack4 transforms
    aligned, _ = align_entries_with_plan(pattern_entries, plan)

    # Build const idx -> slot map
    const_indices = set()
    for op in plan.operations:
        for arg in op.input_vars:
            if 'metadata["relax.expr.Constant"]' in arg:
                idx = match_relax_const_idx(arg)
                if idx is not None:
                    const_indices.add(idx)
    sorted_idx = sorted(const_indices)
    const_idx_to_slot = {const_idx: slot for slot, const_idx in enumerate(sorted_idx)}

    transforms: dict[int, np.ndarray] = {}

    def pack_conv3x3_pack1to4(weight: np.ndarray, out_c: int, in_c: int) -> np.ndarray:
        w = weight.reshape(out_c, in_c, 3, 3)
        outc4 = out_c // 4
        packed = np.zeros((outc4, in_c, 3, 3, 4), dtype=np.float32)
        for oc in range(out_c):
            oc4 = oc // 4
            lane = oc % 4
            packed[oc4, :, :, :, lane] = w[oc, :, :, :]
        return packed.reshape(-1)

    def pack_dw3x3_pack4(weight: np.ndarray, out_c: int) -> np.ndarray:
        if weight.ndim == 4:
            w = weight.reshape(out_c, -1, 3, 3)[:, 0, :, :]
        else:
            w = weight.reshape(out_c, 3, 3)
        outc4 = out_c // 4
        packed = np.zeros((outc4, 3, 3, 4), dtype=np.float32)
        for oc in range(out_c):
            oc4 = oc // 4
            lane = oc % 4
            packed[oc4, :, :, lane] = w[oc, :, :]
        return packed.reshape(-1)

    for entry, op in aligned:
        if entry.get("tvm_op") != "relax.nn.conv2d":
            continue
        ncnn_name = infer_ncnn_function_name(entry_to_pattern_entry(entry))
        if len(op.input_vars) < 2:
            continue
        k_h = int(entry["attrs"].get("kernel_h", 0))
        k_w = int(entry["attrs"].get("kernel_w", 0))
        const_idx = match_relax_const_idx(op.input_vars[1])
        if const_idx is None:
            continue
        slot = const_idx_to_slot.get(const_idx)
        if slot is None:
            continue
        weight_name = weight_order[slot]
        weight = weights[weight_name].astype(np.float32)
        out_c = int(entry["attrs"].get("out_channels", weight.shape[0]))
        in_c = int(entry["attrs"].get("in_channels", weight.shape[1] if weight.ndim > 1 else 1))
        groups = int(entry["attrs"].get("groups", 1))
        if ncnn_name in {"conv3x3s1_pack1to4_neon", "conv3x3s2_pack1to4_neon"}:
            if groups == 1 and k_h == 3 and k_w == 3 and weight.size == out_c * in_c * 9:
                transforms[slot] = pack_conv3x3_pack1to4(weight, out_c, in_c)
        elif ncnn_name in {"convdw3x3s1_pack4_neon", "convdw3x3s2_pack4_neon"}:
            if k_h == 3 and k_w == 3 and weight.size == out_c * 9:
                transforms[slot] = pack_dw3x3_pack4(weight, out_c)

    NcnnWeightPacker.pack(weights, weight_order, output_dir, save_metadata=True, transforms=transforms)
    # Ensure weights.bin is available at output root for validation runner
    weights_bin = output_dir / "lib" / "weights.bin"
    if weights_bin.exists():
        (output_dir / "weights.bin").write_bytes(weights_bin.read_bytes())

    # Emit call_extern lib0.c via TIR build
    mod = build_call_extern_module(pattern_entries)
    rt_mod = tvm.build(mod, target="c")
    lib_dir = output_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    lib0_path = lib_dir / "lib0.c"
    rt_mod.write_to_file(str(lib0_path))

    # Append matched implementations
    sources = collect_neon_sources(matched_symbols)
    write_lib0_with_impl(lib0_path, sources, matched_symbols)
    if NCNN_HEADER.exists():
        (lib_dir / "ncnn.h").write_bytes(NCNN_HEADER.read_bytes())

    # Patch lib0.c to remove TVM header dependencies for standalone build
    if lib0_path.exists():
        lib0_content = lib0_path.read_text(encoding="utf-8", errors="ignore")
        lib0_content = lib0_content.replace('#include "tvm/runtime/base.h"\n', "")
        lib0_content = lib0_content.replace('#include "tvm/runtime/c_backend_api.h"\n', "")
        lib0_content = lib0_content.replace('#include "tvm/ffi/c_api.h"\n', "")
        lib0_content = lib0_content.replace("#include <cstddef>\n", "")
        lib0_path.write_text(lib0_content, encoding="utf-8")

    # Emit main entry
    entry_path = lib_dir / f"{model_name}.c"
    weights_json = output_dir / "metadata" / "weights.json"
    emit_main_entry_from_plan(
        entry_path,
        plan,
        pattern_entries,
        weights_json,
        None,
        dump_ir_tensor_data=dump_ir_tensor_data,
        trace_operator_delay=trace_operator_delay,
    )

    return True

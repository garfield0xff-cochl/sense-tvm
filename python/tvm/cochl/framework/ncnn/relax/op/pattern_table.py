"""Relax DPL patterns, attribute extractors, and ncnn pattern table builder."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional
import re

import tvm
from tvm import relax
from tvm.relax.dpl import DFPattern, is_const, is_op, wildcard

from tvm.cochl import registry


Checker = Callable[[relax.Call], bool]


@dataclass(frozen=True)
class PatternSpec:
    name: str
    pattern: DFPattern
    annotations: Dict[str, DFPattern]
    checker: Optional[Checker] = None


@dataclass(frozen=True)
class PatternEntry:
    tvm_op: str
    ncnn_op: str
    attrs: Dict[str, Any]
    hardware: str
    arch: str


# -------- utils --------

def _as_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, tvm.ir.Array):
        return [int(v) for v in list(value)]
    return [int(value)]


def _get_attr(attrs: Any, name: str, default: Any = None) -> Any:
    if attrs is None:
        return default
    return getattr(attrs, name, default)


def _extract_shape(value: Any) -> Optional[List[int]]:
    if isinstance(value, relax.Constant):
        try:
            return [int(v) for v in value.data.shape]
        except Exception:
            return None
    struct_info = getattr(value, "struct_info", None)
    if struct_info is None:
        return None
    shape = getattr(struct_info, "shape", None)
    if shape is None:
        return None
    if isinstance(shape, relax.ShapeExpr):
        result = []
        for dim in shape.values:
            if isinstance(dim, tvm.tir.IntImm):
                result.append(int(dim.value))
            elif isinstance(dim, int):
                result.append(int(dim))
            else:
                return None
        return result
    if isinstance(shape, tvm.ir.Array):
        result = []
        for dim in shape:
            if isinstance(dim, tvm.tir.IntImm):
                result.append(int(dim.value))
            elif isinstance(dim, int):
                result.append(int(dim))
            else:
                return None
        return result
    return None


def _shape_size(shape: Optional[List[int]]) -> Optional[int]:
    if shape is None:
        return None
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def _broadcast_kind(shape_a: Optional[List[int]], shape_b: Optional[List[int]]) -> str:
    if shape_a is None or shape_b is None:
        return "unknown"
    if shape_a == shape_b:
        return "no_broadcast"
    size_a = _shape_size(shape_a)
    size_b = _shape_size(shape_b)
    if size_a == 1 and size_b != 1:
        return "broadcast_a_scalar"
    if size_b == 1 and size_a != 1:
        return "broadcast_b_scalar"
    if len(shape_a) == len(shape_b):
        b_ok = True
        b_has_one = False
        a_ok = True
        a_has_one = False
        for da, db in zip(shape_a, shape_b):
            if db == 1 and da != 1:
                b_has_one = True
            elif db != da:
                b_ok = False
            if da == 1 and db != 1:
                a_has_one = True
            elif da != db:
                a_ok = False
        if b_ok and b_has_one:
            return "broadcast_b"
        if a_ok and a_has_one:
            return "broadcast_a"
    return "broadcast_unknown"


def _permute_order_type(axes: Optional[List[int]]) -> Optional[int]:
    if axes is None:
        return None
    patterns = {
        (0, 2, 3, 1): 1,
        (0, 3, 1, 2): 2,
        (0, 2, 1, 3): 3,
        (0, 1, 3, 2): 4,
        (0, 1, 2, 3): 0,
        (1, 0, 2, 3): 5,
        (2, 3, 0, 1): 6,
        (3, 1, 2, 0): 7,
        (1, 2, 0, 3): 8,
        (2, 0, 1, 3): 9,
        (3, 0, 1, 2): 10,
        (1, 3, 2, 0): 11,
        (2, 1, 3, 0): 12,
        (3, 2, 0, 1): 13,
        (2, 1, 0, 3): 14,
        (3, 2, 1, 0): 15,
    }
    return patterns.get(tuple(axes), None)


def _conv2d_channel_info(
    call: relax.Call,
    groups: int,
    data_layout: str,
    kernel_layout: str,
):
    data_shape = _extract_shape(call.args[0]) if len(call.args) > 0 else None
    weight_shape = _extract_shape(call.args[1]) if len(call.args) > 1 else None

    if data_layout == "NCHW":
        in_channels = data_shape[1] if data_shape and len(data_shape) > 1 else None
    else:
        in_channels = None

    if kernel_layout == "OIHW":
        out_channels = weight_shape[0] if weight_shape and len(weight_shape) > 0 else None
        kernel_h = weight_shape[2] if weight_shape and len(weight_shape) > 2 else None
        kernel_w = weight_shape[3] if weight_shape and len(weight_shape) > 3 else None
    else:
        out_channels = None
        kernel_h = None
        kernel_w = None

    if in_channels is not None and out_channels is not None:
        pack4_eligible = (in_channels % 4 == 0) and (out_channels % 4 == 0)
    else:
        pack4_eligible = False

    return in_channels, out_channels, kernel_h, kernel_w, pack4_eligible


def _map_binary_op(op_name: str) -> Optional[int]:
    # ncnn BinaryOp operation mapping
    # 0:add 1:sub 2:mul 3:div 4:max 5:min
    if op_name == "add":
        return 0
    if op_name == "multiply":
        return 2
    if op_name == "maximum":
        return 4
    if op_name == "minimum":
        return 5
    return None


# -------- checkers --------

def _check_conv2d(call: relax.Call) -> bool:
    attrs = call.attrs
    data_layout = _get_attr(attrs, "data_layout")
    kernel_layout = _get_attr(attrs, "kernel_layout")
    # permissive defaults to avoid behavior changes
    return data_layout in ("NCHW", None) and kernel_layout in ("OIHW", None)


def _check_binary(call: relax.Call) -> bool:
    base = call.op.name.split(".")[-1]
    return _map_binary_op(base) is not None


def _check_pad(call: relax.Call) -> bool:
    attrs = call.attrs
    pad_mode = _get_attr(attrs, "pad_mode", "constant")
    return pad_mode in ("constant", "edge", "reflect")


# -------- patterns --------

def ncnn_patterns() -> List[PatternSpec]:
    conv = is_op("relax.nn.conv2d")(wildcard(), is_const())
    matmul = is_op("relax.matmul")(wildcard(), wildcard())
    add = is_op("relax.add")(wildcard(), wildcard())
    mul = is_op("relax.multiply")(wildcard(), wildcard())
    maxp = is_op("relax.maximum")(wildcard(), wildcard())
    minp = is_op("relax.minimum")(wildcard(), wildcard())
    pad = is_op("relax.nn.pad")(wildcard())
    concat = is_op("relax.concat")(wildcard())
    perm = is_op("relax.permute_dims")(wildcard())
    squeeze = is_op("relax.squeeze")(wildcard())
    mean = is_op("relax.mean")(wildcard())
    sigmoid = is_op("relax.sigmoid")(wildcard())

    return [
        PatternSpec("relax.nn.conv2d", conv, {"root": conv}, _check_conv2d),
        PatternSpec("relax.matmul", matmul, {"root": matmul}),
        PatternSpec("relax.add", add, {"root": add}, _check_binary),
        PatternSpec("relax.multiply", mul, {"root": mul}, _check_binary),
        PatternSpec("relax.maximum", maxp, {"root": maxp}, _check_binary),
        PatternSpec("relax.minimum", minp, {"root": minp}, _check_binary),
        PatternSpec("relax.nn.pad", pad, {"root": pad}, _check_pad),
        PatternSpec("relax.concat", concat, {"root": concat}),
        PatternSpec("relax.permute_dims", perm, {"root": perm}),
        PatternSpec("relax.squeeze", squeeze, {"root": squeeze}),
        PatternSpec("relax.mean", mean, {"root": mean}),
        PatternSpec("relax.sigmoid", sigmoid, {"root": sigmoid}),
    ]


# -------- entry extraction --------

def build_entry_from_call(call: relax.Call) -> Optional[tuple[str, str, Dict[str, Any]]]:
    """Return (tvm_op, ncnn_op, attrs) for supported calls; None if unsupported."""
    op = call.op
    if not isinstance(op, tvm.ir.Op):
        return None

    op_name = op.name
    attrs = call.attrs
    entry_attrs: Dict[str, Any] = {}
    ncnn_op: Optional[str] = None

    if op_name == "relax.nn.conv2d":
        ncnn_op = "Convolution"
        data_layout = _get_attr(attrs, "data_layout")
        kernel_layout = _get_attr(attrs, "kernel_layout")
        groups = int(_get_attr(attrs, "groups", 1))
        in_ch, out_ch, kh, kw, pack4 = _conv2d_channel_info(
            call, groups, data_layout, kernel_layout
        )
        entry_attrs = {
            "strides": _as_list(_get_attr(attrs, "strides")),
            "padding": _as_list(_get_attr(attrs, "padding")),
            "dilation": _as_list(_get_attr(attrs, "dilation")),
            "groups": groups,
            "data_layout": data_layout,
            "kernel_layout": kernel_layout,
            "out_layout": _get_attr(attrs, "out_layout"),
            "in_channels": in_ch,
            "out_channels": out_ch,
            "kernel_h": kh,
            "kernel_w": kw,
            "pack4_eligible": pack4,
        }
    elif op_name == "relax.matmul":
        ncnn_op = "MatMul"
        entry_attrs = {
            "transpose_a": bool(_get_attr(attrs, "transpose_a", False)),
            "transpose_b": bool(_get_attr(attrs, "transpose_b", False)),
        }
    elif op_name in {"relax.add", "relax.multiply", "relax.maximum", "relax.minimum"}:
        ncnn_op = "BinaryOp"
        base_name = op_name.split(".")[-1]
        lhs_shape = _extract_shape(call.args[0]) if len(call.args) > 0 else None
        rhs_shape = _extract_shape(call.args[1]) if len(call.args) > 1 else None
        broadcast_kind = _broadcast_kind(lhs_shape, rhs_shape)
        entry_attrs = {
            "op_type": _map_binary_op(base_name),
            "op_name": base_name,
            "lhs_shape": lhs_shape,
            "rhs_shape": rhs_shape,
            "broadcast": broadcast_kind,
        }
    elif op_name == "relax.call_tir":
        # Handle lowered binary ops like T_maximum/T_minimum.
        if len(call.args) >= 2 and isinstance(call.args[0], relax.GlobalVar):
            gv = call.args[0]
            base_name = re.sub(r"\d+$", "", gv.name_hint)
            if base_name in {"add", "multiply", "maximum", "minimum"}:
                args_tuple = call.args[1]
                if isinstance(args_tuple, relax.Tuple) and len(args_tuple.fields) >= 2:
                    lhs = args_tuple.fields[0]
                    rhs = args_tuple.fields[1]
                else:
                    lhs = call.args[1]
                    rhs = None
                ncnn_op = "BinaryOp"
                lhs_shape = _extract_shape(lhs) if lhs is not None else None
                rhs_shape = _extract_shape(rhs) if rhs is not None else None
                broadcast_kind = _broadcast_kind(lhs_shape, rhs_shape)
                entry_attrs = {
                    "op_type": _map_binary_op(base_name),
                    "op_name": base_name,
                    "lhs_shape": lhs_shape,
                    "rhs_shape": rhs_shape,
                    "broadcast": broadcast_kind,
                }
                op_name = f"relax.{base_name}"
            else:
                return None
        else:
            return None
    elif op_name == "relax.nn.pad":
        ncnn_op = "Padding"
        entry_attrs = {
            "pad_width": _as_list(_get_attr(attrs, "pad_width")),
            "pad_value": _get_attr(attrs, "pad_value", 0),
            "pad_mode": _get_attr(attrs, "pad_mode", "constant"),
        }
    elif op_name == "relax.concat":
        ncnn_op = "Concat"
        entry_attrs = {
            "axis": int(_get_attr(attrs, "axis", 0)),
        }
    elif op_name == "relax.permute_dims":
        ncnn_op = "Permute"
        axes = _as_list(_get_attr(attrs, "axes"))
        order_type = _permute_order_type(axes)
        entry_attrs = {
            "axes": axes,
            "order_type": order_type,
        }
    elif op_name == "relax.squeeze":
        ncnn_op = "Squeeze"
        entry_attrs = {
            "axis": _as_list(_get_attr(attrs, "axis")),
        }
    elif op_name == "relax.mean":
        ncnn_op = "Reduction"
        entry_attrs = {
            "operation": 3,  # ncnn ReductionOp_MEAN
            "axis": _as_list(_get_attr(attrs, "axis")),
            "keepdims": int(bool(_get_attr(attrs, "keepdims", 0))),
        }
    elif op_name == "relax.sigmoid":
        ncnn_op = "Sigmoid"
        entry_attrs = {}

    if ncnn_op is None:
        return None

    return op_name, ncnn_op, entry_attrs


# -------- packing --------

def _match_pattern(call: relax.Call, patterns: List[PatternSpec], var2val) -> Optional[PatternSpec]:
    for spec in patterns:
        if spec.pattern.match(call, var2val):
            if spec.checker and not spec.checker(call):
                return None
            return spec
    return None


def build_pattern_table(ir_mod: tvm.IRModule, hardware: str) -> List[Dict[str, Any]]:
    """Build mapping table from Relax ops to ncnn operators."""
    entries: List[PatternEntry] = []
    arch = registry.get_architecture(hardware)
    patterns = ncnn_patterns()

    def _visit_func(func: relax.Function):
        var2val = relax.analysis.get_var2val(func)

        def _visit(expr):
            if isinstance(expr, relax.Call):
                op_name = None
                if isinstance(expr.op, tvm.ir.Op):
                    op_name = expr.op.name
                if op_name != "relax.call_tir":
                    spec = _match_pattern(expr, patterns, var2val)
                    if spec is None:
                        return
                result = build_entry_from_call(expr)
                if result is None:
                    return
                op_name_use, ncnn_op, entry_attrs = result
                entries.append(
                    PatternEntry(
                        tvm_op=op_name_use,
                        ncnn_op=ncnn_op,
                        attrs=entry_attrs,
                        hardware=hardware,
                        arch=arch,
                    )
                )

        relax.analysis.post_order_visit(func, _visit)

    if "main" in ir_mod.functions:
        main_func = ir_mod["main"]
        if isinstance(main_func, relax.Function):
            _visit_func(main_func)
    else:
        for func in ir_mod.functions.values():
            if isinstance(func, relax.Function):
                _visit_func(func)

    return [asdict(e) for e in entries]


# -------- ncnn name inference --------

def infer_ncnn_function_name(entry: PatternEntry) -> str:
    """Infer a deterministic ncnn function name from a pattern entry."""
    tvm_op = entry.tvm_op.split(".")[-1]
    attrs = entry.attrs or {}

    if tvm_op == "permute_dims":
        order_type = attrs.get("order_type")
        if order_type is not None:
            return f"permute_order_{order_type}"
        return "Permute"
    if tvm_op == "conv2d":
        kh = attrs.get("kernel_h")
        kw = attrs.get("kernel_w")
        strides = attrs.get("strides") or []
        dilation = attrs.get("dilation") or []
        groups = attrs.get("groups", 1)
        pack4 = bool(attrs.get("pack4_eligible", False))
        if (
            kh == 3
            and kw == 3
            and strides == [1, 1]
            and dilation == [1, 1]
            and attrs.get("in_channels") == groups
            and attrs.get("out_channels") == groups
        ):
            if pack4:
                return "convdw3x3s1_pack4_neon"
            return "convdw3x3s1_neon"
        if (
            kh == 3
            and kw == 3
            and strides == [2, 2]
            and dilation == [1, 1]
            and attrs.get("in_channels") == groups
            and attrs.get("out_channels") == groups
            and pack4
        ):
            return "convdw3x3s2_pack4_neon"
        if (
            kh == 3
            and kw == 3
            and strides == [2, 2]
            and dilation == [1, 1]
            and pack4
        ):
            return "conv3x3s2_pack1to4_neon"
        if (
            kh == 3
            and kw == 3
            and strides == [2, 2]
            and dilation == [1, 1]
            and pack4
            and groups == 1
        ):
            return "conv3x3s2_packed_neon"
        if (
            kh == 3
            and kw == 3
            and strides == [2, 2]
            and dilation == [1, 1]
            and groups == 1
        ):
            return "conv3x3s2_neon"
        if (
            kh == 1
            and kw == 1
            and strides == [1, 1]
            and dilation == [1, 1]
            and groups == 1
        ):
            return "conv1x1s1_neon"
        if (
            kh == 3
            and kw == 3
            and strides == [1, 1]
            and dilation == [1, 1]
            and groups == 1
            and pack4
        ):
            return "conv3x3s1_pack1to4_neon"
        return "Convolution"
    if tvm_op in {"add", "multiply", "maximum", "minimum"}:
        if attrs.get("broadcast") == "no_broadcast":
            return "binary_op_vector_no_broadcast"
        if attrs.get("broadcast") in {"broadcast_b_scalar", "broadcast_b"}:
            return "binary_op_vector_broadcast_b"
        if attrs.get("broadcast") in {"broadcast_a_scalar", "broadcast_a"}:
            return "binary_op_vector_broadcast_a"
        return "binary_op_broadcast"
    if tvm_op == "pad":
        return "Padding"
    if tvm_op == "concat":
        return "Concat"
    if tvm_op == "matmul":
        ta = int(bool(attrs.get("transpose_a", False)))
        tb = int(bool(attrs.get("transpose_b", False)))
        return f"matmul_ta{ta}_tb{tb}"
    if tvm_op == "mean":
        keep = int(bool(attrs.get("keepdims", 0)))
        axis = attrs.get("axis") or []
        if axis:
            axis_str = "x".join(str(a) for a in axis)
        else:
            axis_str = "all"
        return f"reduction_mean_{axis_str}_k{keep}"
    if tvm_op == "squeeze":
        return "Squeeze"
    if tvm_op == "sigmoid":
        return "sigmoid"
    return "Custom"

"""Fuse pad -> conv2d by absorbing padding into conv attrs when possible."""

from typing import Optional, Sequence

import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, mutator


def _call_tir_callee(call: relax.Call):
    if not isinstance(call.op, tvm.ir.Op):
        return None
    if call.op.name != "relax.call_tir":
        return None
    if not call.args:
        return None
    return call.args[0]


def _call_tir_args(call: relax.Call):
    if len(call.args) < 2:
        return []
    args = call.args[1]
    if isinstance(args, relax.Tuple):
        return list(args.fields)
    if isinstance(args, (list, tuple)):
        return list(args)
    return []


def _extract_pad_hw(mod: tvm.IRModule, callee) -> Optional[Sequence[int]]:
    if not isinstance(callee, tvm.ir.GlobalVar):
        return None
    if callee.name_hint is None or not callee.name_hint.startswith("pad"):
        return None
    primfunc = mod[callee]
    if not isinstance(primfunc, tvm.tir.PrimFunc):
        return None
    if len(primfunc.params) < 2:
        return None
    buf_map = primfunc.buffer_map
    in_buf = buf_map.get(primfunc.params[0])
    out_buf = buf_map.get(primfunc.params[1])
    if in_buf is None or out_buf is None:
        return None
    in_shape = list(in_buf.shape)
    out_shape = list(out_buf.shape)
    if len(in_shape) != 4 or len(out_shape) != 4:
        return None
    try:
        in_h, in_w = int(in_shape[2]), int(in_shape[3])
        out_h, out_w = int(out_shape[2]), int(out_shape[3])
    except Exception:
        return None
    pad_h = out_h - in_h
    pad_w = out_w - in_w
    if pad_h < 0 or pad_w < 0:
        return None
    # Assume padding is only added to bottom/right (common in generated pad)
    return [0, 0, pad_h, pad_w]


@mutator
class _FusePadConvMutator(PyExprMutator):
    def __init__(self, mod: tvm.IRModule):
        super().__init__(mod)
        self._mod = mod
        self.matched = 0

    def visit_call_(self, call):  # pylint: disable=arguments-differ
        call = super().visit_call_(call)
        if not isinstance(call, relax.Call):
            return call
        if not isinstance(call.op, tvm.ir.Op):
            return call
        if call.op.name != "relax.nn.conv2d":
            return call

        if not call.args:
            return call
        data = call.args[0]
        if not isinstance(data, relax.Call):
            return call
        callee = _call_tir_callee(data)
        if callee is None:
            return call
        pad_args = _call_tir_args(data)
        if len(pad_args) != 1:
            return call
        padding = _extract_pad_hw(self._mod, callee)
        if padding is None:
            return call

        attrs = call.attrs
        self.matched += 1
        return relax.op.nn.conv2d(
            pad_args[0],
            call.args[1],
            strides=list(getattr(attrs, "strides", [1, 1])),
            padding=padding,
            dilation=list(getattr(attrs, "dilation", [1, 1])),
            groups=int(getattr(attrs, "groups", 1)),
            data_layout=str(getattr(attrs, "data_layout", "NCHW")),
            kernel_layout=str(getattr(attrs, "kernel_layout", "OIHW")),
            out_layout=str(getattr(attrs, "out_layout", "NCHW")),
            out_dtype=str(getattr(attrs, "out_dtype", "")),
        )


@relax.transform.function_pass(opt_level=0)
class FusePadConv:
    """Fuse pad->conv2d when padding can be inferred from TIR pad shape."""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        mut = _FusePadConvMutator(mod)
        out = mut.visit_expr(func)
        # silent by default
        return out

"""Fuse maximum/minimum clip pattern into a single clip op."""

from typing import Optional

import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator, mutator


def _call_tir_name(call: relax.Call) -> Optional[str]:
    if not isinstance(call.op, tvm.ir.Op):
        return None
    if call.op.name != "relax.call_tir":
        return None
    if not call.args:
        return None
    callee = call.args[0]
    return getattr(callee, "name_hint", None)


def _call_tir_args(call: relax.Call):
    if len(call.args) < 2:
        return []
    args = call.args[1]
    if isinstance(args, relax.Tuple):
        return list(args.fields)
    if isinstance(args, (list, tuple)):
        return list(args)
    return []


def _is_scalar_const(expr: relax.Expr, value: float) -> bool:
    if not isinstance(expr, relax.expr.Constant):
        return False
    try:
        data = expr.data.numpy()
        if data.size != 1:
            return False
        return float(data.item()) == float(value)
    except Exception:
        return False


@mutator
class _FuseRelu6Mutator(PyExprMutator):
    def __init__(self):
        super().__init__()
        self.matched = 0

    def visit_call_(self, call):  # pylint: disable=arguments-differ
        call = super().visit_call_(call)
        if not isinstance(call, relax.Call):
            return call

        name = _call_tir_name(call)
        if not name or not name.startswith("minimum"):
            return call

        args = _call_tir_args(call)
        if len(args) != 2 or not _is_scalar_const(args[1], 6.0):
            return call

        inner = args[0]
        if not isinstance(inner, relax.Call):
            return call
        inner_name = _call_tir_name(inner)
        if not inner_name or not inner_name.startswith("maximum"):
            return call
        inner_args = _call_tir_args(inner)
        if len(inner_args) != 2 or not _is_scalar_const(inner_args[1], 0.0):
            return call

        self.matched += 1
        return relax.op.clip(inner_args[0], 0.0, 6.0)


@relax.transform.function_pass(opt_level=0)
class FuseRelu6:
    """Fuse maximum/minimum clip pattern into a single clip op."""

    def transform_function(self, func, mod, pass_ctx):  # pylint: disable=unused-argument
        mut = _FuseRelu6Mutator()
        out = mut.visit_expr(func)
        # silent by default
        return out

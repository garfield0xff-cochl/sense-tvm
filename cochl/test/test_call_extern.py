#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import tvm
import importlib
from tvm.script import ir as I
from tvm.script import tir as T
from tvm import relax, ir as _ir
from tvm.relax.expr_functor import PyExprVisitor, visitor

from tvm.cochl.core.translate import translate_onnx


@visitor
class Conv2DPatternVisitor(PyExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.matches = []

    def visit_call_(self, call):  # type: ignore[override]
        if hasattr(call, "op") and isinstance(call.op, _ir.Op):
            if call.op.name == "relax.nn.conv2d":
                attrs = call.attrs
                if attrs is not None:
                    def _to_list(x):
                        if x is None:
                            return None
                        try:
                            return list(x)
                        except Exception:
                            return x

                    kernel_size = _to_list(getattr(attrs, "kernel_size", None))
                    if kernel_size is None:
                        try:
                            w_arg = call.args[1]
                            if isinstance(w_arg.struct_info, relax.TensorStructInfo):
                                w_shape = w_arg.struct_info.shape
                                if w_shape is not None and len(w_shape) >= 4:
                                    # assume OIHW
                                    kernel_size = [int(w_shape[2]), int(w_shape[3])]
                        except Exception:
                            pass
                    strides = _to_list(getattr(attrs, "strides", None))
                    padding = _to_list(getattr(attrs, "padding", None))
                    data_layout = getattr(attrs, "data_layout", None)
                    kernel_layout = getattr(attrs, "kernel_layout", None)
                    out_layout = getattr(attrs, "out_layout", None)
                    dilation = _to_list(getattr(attrs, "dilation", None))
                    groups = getattr(attrs, "groups", None)

                    entry = {
                        "kernel_size": kernel_size,
                        "strides": strides,
                        "padding": padding,
                        "data_layout": data_layout,
                        "kernel_layout": kernel_layout,
                        "out_layout": out_layout,
                        "dilation": dilation,
                        "groups": groups,
                    }
                    self.matches.append(entry)
        return super().visit_call_(call)


def _has_stride(v: Conv2DPatternVisitor, stride: int) -> bool:
    for m in v.matches:
        if (
            m.get("kernel_size") == [3, 3]
            and m.get("strides") == [stride, stride]
            and m.get("padding") == [0, 0, 0, 0]
            and m.get("data_layout") == "NCHW"
        ):
            return True
    return False


def _build_tir_module(
    in_c: int, in_h: int, in_w: int, out_c: int, out_h: int, out_w: int, symbol: str
):
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.handle, W: T.handle, B: T.handle, O: T.handle):
            T.func_attr({"tir.noalias": True})
            A_buf = T.match_buffer(A, (in_c, in_h, in_w), dtype="float32")
            W_buf = T.match_buffer(W, (out_c, in_c, 3, 3, 4), dtype="float32")
            B_buf = T.match_buffer(B, (out_c, 4), dtype="float32")
            O_buf = T.match_buffer(O, (out_c, out_h, out_w, 4), dtype="float32")
            T.evaluate(
                T.call_extern(
                    "int32",
                    symbol,
                    A_buf.data,
                    T.int32(in_c),
                    T.int32(in_h),
                    T.int32(in_w),
                    W_buf.data,
                    B_buf.data,
                    O_buf.data,
                    T.int32(out_c),
                    T.int32(out_h),
                    T.int32(out_w),
                )
            )

    return Module


def _append_impl_to_lib0(lib0_path: Path, impl_path: Path, symbol: str) -> bool:
    if not lib0_path.exists() or not impl_path.exists():
        return False
    text = lib0_path.read_text(encoding="utf-8", errors="ignore")
    marker = f"/* COCHL_EXTERN_IMPL_BEGIN {symbol} */"
    if marker in text:
        return False

    proto = (
        f"/* COCHL_EXTERN_DECL {symbol} */\n"
        f"int32_t {symbol}(const float* input, int in_c, int in_h, int in_w, "
        f"const float* kernel, const float* bias, float* output, int out_c, int out_h, int out_w);\n\n"
    )
    text = proto + text
    impl = impl_path.read_text(encoding="utf-8", errors="ignore")
    text += f"\n\n{marker}\n{impl}\n/* COCHL_EXTERN_IMPL_END */\n"
    lib0_path.write_text(text, encoding="utf-8")
    return True


def _attach_tir_target(mod: tvm.IRModule, target: str) -> tvm.IRModule:
    tgt = tvm.target.Target(target)
    new_funcs = {}
    for gv, func in mod.functions.items():
        if hasattr(func, "with_attr"):
            func = func.with_attr("target", tgt)
        new_funcs[gv] = func
    return tvm.IRModule(new_funcs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("sense/onnx/model_main_17.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--lib0",
        type=Path,
        default=Path("cochl/test/lib/lib0.c"),
        help="Path to output lib0.c",
    )
    args = parser.parse_args()

    ir_mod, _, _, _, _ = translate_onnx(args.onnx)
    v = Conv2DPatternVisitor()
    for _, func in ir_mod.functions.items():
        if isinstance(func, relax.Function):
            v.visit_expr(func)

    matched_s1 = _has_stride(v, 1)
    matched_s2 = _has_stride(v, 2)

    in_c, out_c = 32, 16
    out_h, out_w = 56, 56
    if matched_s2:
        in_h, in_w = out_h * 2 + 1, out_w * 2 + 1
        symbol = "cochl_conv3x3s2_pack1to4_neon"
    else:
        in_h, in_w = out_h + 2, out_w + 2
        symbol = "cochl_conv3x3s1_pack1to4_neon"

    mod = _build_tir_module(in_c, in_h, in_w, out_c, out_h, out_w, symbol)
    mod = _attach_tir_target(mod, "c")
    relax_pass = importlib.import_module("tvm.cochl.framework.tvm_c.relax.pass")
    mod = relax_pass.get_unpacked_passes()(mod)
    rt_mod = tvm.build(mod, target="c")

    out_dir = args.lib0.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rt_mod.write_to_file(str(args.lib0))

    if matched_s1:
        impl_path = Path("cochl/src/target/neon/nchw/conv_3x3_pack1to4_impl.c")
        _append_impl_to_lib0(args.lib0, impl_path, "cochl_conv3x3s1_pack1to4_neon")
    if matched_s2:
        impl_path = Path("cochl/src/target/neon/nchw/conv_3x3_pack1to4_s2_impl.c")
        _append_impl_to_lib0(args.lib0, impl_path, "cochl_conv3x3s2_pack1to4_neon")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

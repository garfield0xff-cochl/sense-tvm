import tvm
from tvm import tir
from tvm.script import tir as T


def relu6_nhwc(name, batch, channels, height, width):
    @T.prim_func
    def relu6_func(
        Input: T.Buffer((batch, height, width, channels), "float32"),  # NHWC
        Output: T.Buffer((batch, height, width, channels), "float32")  # NHWC
    ):
        T.func_attr({"global_symbol": name, "tir.noalias": True})

        for n, h, w, c in T.grid(batch, height, width, channels):
            with T.sblock("relu6"):
                vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
                T.reads(Input[vn, vh, vw, vc])
                T.writes(Output[vn, vh, vw, vc])
                val: T.float32 = Input[vn, vh, vw, vc]
                Output[vn, vh, vw, vc] = T.min(T.max(val, T.float32(0.0)), T.float32(6.0))

    return relu6_func

def add_nhwc(name, batch, channels, height, width):
    @T.prim_func
    def add_func(
        A: T.Buffer((batch, height, width, channels), "float32"),  # NHWC
        B: T.Buffer((batch, height, width, channels), "float32"),  # NHWC
        Output: T.Buffer((batch, height, width, channels), "float32")  # NHWC
    ):
        T.func_attr({"global_symbol": name, "tir.noalias": True})

        for n, h, w, c in T.grid(batch, height, width, channels):
            with T.sblock("add"):
                vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
                T.reads(A[vn, vh, vw, vc], B[vn, vh, vw, vc])
                T.writes(Output[vn, vh, vw, vc])
                Output[vn, vh, vw, vc] = A[vn, vh, vw, vc] + B[vn, vh, vw, vc]

    return add_func

import tvm
from tvm import tir
from tvm.script import tir as T


def concat_nhwc(name, batch, height, width, in_ch1, in_ch2, in_ch3):
    out_ch = in_ch1 + in_ch2 + in_ch3

    @T.prim_func
    def concat_func(
        Input1: T.Buffer((batch, height, width, in_ch1), "float32"),
        Input2: T.Buffer((batch, height, width, in_ch2), "float32"),
        Input3: T.Buffer((batch, height, width, in_ch3), "float32"),
        Output: T.Buffer((batch, height, width, out_ch), "float32")
    ):
        T.func_attr({"global_symbol": name, "tir.noalias": True})

        # Copy first input
        for n, h, w, c in T.grid(batch, height, width, in_ch1):
            with T.sblock("concat1"):
                vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
                T.reads(Input1[vn, vh, vw, vc])
                T.writes(Output[vn, vh, vw, vc])
                Output[vn, vh, vw, vc] = Input1[vn, vh, vw, vc]

        # Copy second input
        for n, h, w, c in T.grid(batch, height, width, in_ch2):
            with T.sblock("concat2"):
                vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
                T.reads(Input2[vn, vh, vw, vc])
                T.writes(Output[vn, vh, vw, in_ch1 + vc])
                Output[vn, vh, vw, in_ch1 + vc] = Input2[vn, vh, vw, vc]

        # Copy third input
        for n, h, w, c in T.grid(batch, height, width, in_ch3):
            with T.sblock("concat3"):
                vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
                T.reads(Input3[vn, vh, vw, vc])
                T.writes(Output[vn, vh, vw, in_ch1 + in_ch2 + vc])
                Output[vn, vh, vw, in_ch1 + in_ch2 + vc] = Input3[vn, vh, vw, vc]

    return concat_func
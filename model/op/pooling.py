import tvm
from tvm import tir
from tvm.script import tir as T


def global_avg_pool_nhwc(name, batch, channels, height, width):
    total_size = height * width

    @T.prim_func
    def global_pool_func(
        Input: T.Buffer((batch, height, width, channels), "float32"),  # NHWC
        Output: T.Buffer((batch, channels), "float32")
    ):
        T.func_attr({"global_symbol": name, "tir.noalias": True})

        # Sum reduction
        for n, c, h, w in T.grid(batch, channels, height, width):
            with T.sblock("global_pool"):
                vn, vc, vh, vw = T.axis.remap("SSRR", [n, c, h, w])
                T.reads(Input[vn, vh, vw, vc])
                T.writes(Output[vn, vc])
                with T.init():
                    Output[vn, vc] = T.float32(0.0)
                Output[vn, vc] = Output[vn, vc] + Input[vn, vh, vw, vc]

        # Average
        for n, c in T.grid(batch, channels):
            with T.sblock("pool_div"):
                vn, vc = T.axis.remap("SS", [n, c])
                T.reads(Output[vn, vc])
                T.writes(Output[vn, vc])
                Output[vn, vc] = Output[vn, vc] / T.float32(total_size)

    return global_pool_func
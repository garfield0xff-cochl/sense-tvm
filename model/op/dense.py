import tvm
from tvm import tir
from tvm.script import tir as T


def linear(name, batch, in_features, out_features):
    @T.prim_func
    def linear_func(
        Input: T.Buffer((batch, in_features), "float32"),
        Weight: T.Buffer((out_features, in_features), "float32"),
        Bias: T.Buffer((out_features,), "float32"),
        Output: T.Buffer((batch, out_features), "float32")
    ):
        T.func_attr({"global_symbol": name, "tir.noalias": True})

        for n, o, i in T.grid(batch, out_features, in_features):
            with T.sblock("linear"):
                vn, vo, vi = T.axis.remap("SSR", [n, o, i])
                T.reads(Bias[vo], Input[vn, vi], Weight[vo, vi])
                T.writes(Output[vn, vo])
                with T.init():
                    Output[vn, vo] = Bias[vo]
                Output[vn, vo] = Output[vn, vo] + Input[vn, vi] * Weight[vo, vi]

    return linear_func
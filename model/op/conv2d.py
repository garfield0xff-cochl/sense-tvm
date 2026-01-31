import tvm
from tvm import tir
from tvm.script import tir as T


def conv2d_nhwc(name, batch, in_ch, in_h, in_w, out_ch, out_h, out_w, kernel, stride, padding):
    pad_h = in_h + 2 * padding
    pad_w = in_w + 2 * padding

    @T.prim_func
    def conv2d(
        Input: T.Buffer((batch, in_h, in_w, in_ch), "float32"),  # NHWC
        Weight: T.Buffer((out_ch, in_ch, kernel, kernel), "float32"),  # OIHW
        Bias: T.Buffer((out_ch,), "float32"),
        Output: T.Buffer((batch, out_h, out_w, out_ch), "float32")  # NHWC
    ):
    
        T.func_attr({"global_symbol": name, "tir.noalias": True})
        PaddedInput = T.alloc_buffer((batch, pad_h, pad_w, in_ch), "float32")

        for n, h, w, c in T.grid(batch, pad_h, pad_w, in_ch):
            with T.sblock("pad"):
                vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
                PaddedInput[vn, vh, vw, vc] = T.if_then_else(
                    T.likely(vh >= padding and vh < in_h + padding and
                            vw >= padding and vw < in_w + padding),
                    Input[vn, vh - padding, vw - padding, vc],
                    T.float32(0.0)
                )

        for n, h, w, co, ci, kh, kw in T.grid(batch, out_h, out_w, out_ch, in_ch, kernel, kernel):
            with T.sblock("conv"):
                # SSSS : Output[vn, vh, vw, vco]  
                # RR   : ci, kh, kw -> reduction 
                vn, vh, vw, vco, vci, vkh, vkw = T.axis.remap("SSSSRRR", [n, h, w, co, ci, kh, kw])
                T.reads(Bias[vco], PaddedInput[vn, vh * stride + vkh, vw * stride + vkw, vci], Weight[vco, vci, vkh, vkw])
                T.writes(Output[vn, vh, vw, vco])
                with T.init():
                    Output[vn, vh, vw, vco] = Bias[vco]
                Output[vn, vh, vw, vco] = Output[vn, vh, vw, vco] + \
                    PaddedInput[vn, vh * stride + vkh, vw * stride + vkw, vci] * Weight[vco, vci, vkh, vkw]

    return conv2d


def depthwise_conv2d_nhwc(name, batch, channels, in_h, in_w, out_h, out_w, kernel, stride, padding):
    pad_h = in_h + 2 * padding
    pad_w = in_w + 2 * padding

    @T.prim_func
    def depthwise_func(
        Input: T.Buffer((batch, in_h, in_w, channels), "float32"),  # NHWC
        Weight: T.Buffer((channels, 1, kernel, kernel), "float32"),  # OIHW
        Bias: T.Buffer((channels,), "float32"),
        Output: T.Buffer((batch, out_h, out_w, channels), "float32")  # NHWC
    ):
        T.func_attr({"global_symbol": name, "tir.noalias": True})
        PaddedInput = T.alloc_buffer((batch, pad_h, pad_w, channels), "float32")

        # Padding
        for n, h, w, c in T.grid(batch, pad_h, pad_w, channels):
            with T.sblock("pad"):
                vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
                PaddedInput[vn, vh, vw, vc] = T.if_then_else(
                    T.likely(vh >= padding and vh < in_h + padding and
                            vw >= padding and vw < in_w + padding),
                    Input[vn, vh - padding, vw - padding, vc],
                    T.float32(0.0)
                )

        # Depthwise convolution
        for n, h, w, c, kh, kw in T.grid(batch, out_h, out_w, channels, kernel, kernel):
            with T.sblock("depthwise"):
                vn, vh, vw, vc, vkh, vkw = T.axis.remap("SSSSRR", [n, h, w, c, kh, kw])
                T.reads(Bias[vc], PaddedInput[vn, vh * stride + vkh, vw * stride + vkw, vc], Weight[vc, 0, vkh, vkw])
                T.writes(Output[vn, vh, vw, vc])
                with T.init():
                    Output[vn, vh, vw, vc] = Bias[vc]
                Output[vn, vh, vw, vc] = Output[vn, vh, vw, vc] + \
                    PaddedInput[vn, vh * stride + vkh, vw * stride + vkw, vc] * Weight[vc, 0, vkh, vkw]

    return depthwise_func



        


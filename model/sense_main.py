'''
This model is designed for two sub mobilenetv2 Model
2026-01-27 kimgyujin
'''

import tvm
from tvm import tir
from tvm.script import tir as T
import sys
import os

# Support both module import and direct script execution
if __name__ == "__main__":
    # Running as script from model directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.op.conv2d import conv2d_nhwc, depthwise_conv2d_nhwc
    from model.op.elemwise import relu6_nhwc, add_nhwc
    from model.op.pooling import global_avg_pool_nhwc
    from model.op.dense import linear
    from model.op.concat import concat_nhwc
    from model.parser import parse_onnx_layers
else:
    # Imported as module
    from .op.conv2d import conv2d_nhwc, depthwise_conv2d_nhwc
    from .op.elemwise import relu6_nhwc, add_nhwc
    from .op.pooling import global_avg_pool_nhwc
    from .op.dense import linear
    from .op.concat import concat_nhwc
    from .parser import parse_onnx_layers


def sense_sub_model1(tir_funcs, onnx_path):
    """Build m_main_1 submodel (MobileNetV2 variant with 24 initial channels)"""
    input_shape, output_shape, graph, weights = parse_onnx_layers(onnx_path)

    batch, in_h, in_w, in_ch = input_shape  # [1, 128, 192, 1]
    num_classes = output_shape[1]  # 863

    # test for sense sdk input shape (1 channel)
    H, W = in_h, in_w

    H_out, W_out = 64, 96
    print(f"  create concat_input (1→3 channels)...")
    tir_funcs["concat_input"] = concat_nhwc("concat_input", 1, H, W, 1, 1, 1)
    print(f"  create first_conv (3→24, 128x192→64x96)...")
    tir_funcs["first_conv"] = conv2d_nhwc("first_conv", 1, 3, H, W, 24, H_out, W_out, 3, 2, 0)
    print(f"  create first_relu6...")
    tir_funcs["first_relu6"] = relu6_nhwc("first_relu6", 1, 24, H_out, W_out)

    inverted_residual_setting = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 24, 3, 2],
        [6, 48, 4, 2],
        [6, 72, 3, 1],
        [6, 120, 3, 2],
        [6, 240, 1, 1],
    ]

    H, W = H_out, W_out
    current_ch = 24

    block_idx = 0

    for t, out_ch, n, s in inverted_residual_setting:
        for i in range(n):
            stride = s if i == 0 else 1
            hidden_ch = int(current_ch * t)

            if stride == 2:
                H_out, W_out = H // 2, W // 2
            else:
                H_out, W_out = H, W

            # Expansion (if expand_ratio != 1)
            if t != 1:
                expand_name = f"block{block_idx}_expand"
                print(f"  create {expand_name} ({current_ch}→{hidden_ch})...")
                tir_funcs[expand_name] = conv2d_nhwc(expand_name, 1, current_ch, H, W, hidden_ch, H, W, 1, 1, 0)

                expand_relu_name = f"block{block_idx}_expand_relu6"
                tir_funcs[expand_relu_name] = relu6_nhwc(expand_relu_name, 1, hidden_ch, H, W)

            # Depthwise
            dw_name = f"block{block_idx}_dw"
            print(f"  create {dw_name} ({hidden_ch}ch, {H}x{W}→{H_out}x{W_out})...")
            tir_funcs[dw_name] = depthwise_conv2d_nhwc(dw_name, 1, hidden_ch, H, W, H_out, W_out, 3, stride, 1)

            dw_relu_name = f"block{block_idx}_dw_relu6"
            tir_funcs[dw_relu_name] = relu6_nhwc(dw_relu_name, 1, hidden_ch, H_out, W_out)

            # projection
            proj_name = f"block{block_idx}_project"
            print(f"  create {proj_name} ({hidden_ch}→{out_ch})...")
            tir_funcs[proj_name] = conv2d_nhwc(proj_name, 1, hidden_ch, H_out, W_out, out_ch, H_out, W_out, 1, 1, 0)

            # residual add
            if stride == 1 and current_ch == out_ch:
                add_name = f"block{block_idx}_add"
                print(f"  create {add_name} (residual)...")
                tir_funcs[add_name] = add_nhwc(add_name, 1, out_ch, H_out, W_out)

            H, W = H_out, W_out
            current_ch = out_ch
            block_idx += 1

    # last conv
    print(f"  create last_conv ({current_ch}→1280)...")
    tir_funcs["last_conv"] = conv2d_nhwc("last_conv", 1, current_ch, H, W, 1280, H, W, 1, 1, 0)
    print(f"  create last_relu6...")
    tir_funcs["last_relu6"] = relu6_nhwc("last_relu6", 1, 1280, H, W)

    # avg pool
    print(f"  create global_pool ({H}x{W}→1x1)...")
    tir_funcs["global_pool"] = global_avg_pool_nhwc("global_pool", 1, 1280, H, W)

    # classifier
    print(f"  create m1_classifier (1280→{num_classes})...")
    tir_funcs["m1_classifier"] = linear("m1_classifier", 1, 1280, num_classes)

    print()
    print(f"  m_main_1: built")
    print()

    return num_classes

def sense_sub_model0(tir_funcs, onnx_path, num_classes):
    """Build m_main_0 submodel (MobileNetV2 variant with 32 initial channels)"""
    input_shape, output_shape, graph, weights = parse_onnx_layers(onnx_path)

    batch, in_h, in_w, in_ch = input_shape  # [1, 128, 192, 1]

    H, W = in_h, in_w
    H_out, W_out = 64, 96
    print(f"  create m0_concat_input (1→3 channels)...")
    tir_funcs["m0_concat_input"] = concat_nhwc("m0_concat_input", 1, H, W, 1, 1, 1)
    print(f"  create m0_first_conv (3→32, 128x192→64x96)...")
    tir_funcs["m0_first_conv"] = conv2d_nhwc("m0_first_conv", 1, 3, H, W, 32, H_out, W_out, 3, 2, 1)
    print(f"  create m0_first_relu6...")
    tir_funcs["m0_first_relu6"] = relu6_nhwc("m0_first_relu6", 1, 32, H_out, W_out)

    m0_inverted_residual_setting = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    H, W = H_out, W_out
    current_ch = 32

    block_idx = 0

    for t, out_ch, n, s in m0_inverted_residual_setting:
        for i in range(n):
            stride = s if i == 0 else 1
            hidden_ch = int(current_ch * t)

            if stride == 2:
                H_out, W_out = H // 2, W // 2
            else:
                H_out, W_out = H, W

            # Expansion (if expand_ratio != 1)
            if t != 1:
                expand_name = f"m0_block{block_idx}_expand"
                print(f"  create {expand_name} ({current_ch}→{hidden_ch})...")
                tir_funcs[expand_name] = conv2d_nhwc(expand_name, 1, current_ch, H, W, hidden_ch, H, W, 1, 1, 0)

                expand_relu_name = f"m0_block{block_idx}_expand_relu6"
                tir_funcs[expand_relu_name] = relu6_nhwc(expand_relu_name, 1, hidden_ch, H, W)

            # Depthwise
            dw_name = f"m0_block{block_idx}_dw"
            print(f"  create {dw_name} ({hidden_ch}ch, {H}x{W}→{H_out}x{W_out})...")
            tir_funcs[dw_name] = depthwise_conv2d_nhwc(dw_name, 1, hidden_ch, H, W, H_out, W_out, 3, stride, 1)

            dw_relu_name = f"m0_block{block_idx}_dw_relu6"
            tir_funcs[dw_relu_name] = relu6_nhwc(dw_relu_name, 1, hidden_ch, H_out, W_out)

            # projection
            proj_name = f"m0_block{block_idx}_project"
            print(f"  create {proj_name} ({hidden_ch}→{out_ch})...")
            tir_funcs[proj_name] = conv2d_nhwc(proj_name, 1, hidden_ch, H_out, W_out, out_ch, H_out, W_out, 1, 1, 0)

            # residual add
            if stride == 1 and current_ch == out_ch:
                add_name = f"m0_block{block_idx}_add"
                print(f"  create {add_name} (residual)...")
                tir_funcs[add_name] = add_nhwc(add_name, 1, out_ch, H_out, W_out)

            H, W = H_out, W_out
            current_ch = out_ch
            block_idx += 1

    # last conv
    print(f"  create m0_last_conv ({current_ch}→1280)...")
    tir_funcs["m0_last_conv"] = conv2d_nhwc("m0_last_conv", 1, current_ch, H, W, 1280, H, W, 1, 1, 0)
    print(f"  create m0_last_relu6...")
    tir_funcs["m0_last_relu6"] = relu6_nhwc("m0_last_relu6", 1, 1280, H, W)

    # avg pool
    print(f"  create m0_global_pool ({H}x{W}→1x1)...")
    tir_funcs["m0_global_pool"] = global_avg_pool_nhwc("m0_global_pool", 1, 1280, H, W)

    # classifier
    print(f"  create m0_classifier (1280→{num_classes})...")
    tir_funcs["m0_classifier"] = linear("m0_classifier", 1, 1280, num_classes)

    print()
    print(f"  m_main_0: built")
    print()


def ensemble(tir_funcs, num_classes):
    """Build ensemble averaging function for combining two model outputs"""
    print(f"  create ensemble_avg...")

    @T.prim_func
    def ensemble_avg_func(
        M1_Output: T.Buffer((1, num_classes), "float32"),
        M0_Output: T.Buffer((1, num_classes), "float32"),
        Output: T.Buffer((1, num_classes), "float32")
    ):
        T.func_attr({"global_symbol": "ensemble_avg", "tir.noalias": True})
        for n, c in T.grid(1, num_classes):
            with T.sblock("avg"):
                vn, vc = T.axis.remap("SS", [n, c])
                T.reads(M1_Output[vn, vc], M0_Output[vn, vc])
                T.writes(Output[vn, vc])
                Output[vn, vc] = (M1_Output[vn, vc] + M0_Output[vn, vc]) / T.float32(2.0)

    tir_funcs["ensemble_avg"] = ensemble_avg_func

    print()
    print(f"  ensemble: built")
    print()


# TODO: schedule operation for rasp2
# vectorize for simd 128bit register [x]
# parallel for 4 thread              [x]
# select 4 thread loop target        [x]

def schedule_op_for_rasp2(mod):
    """Apply parallelization schedules for Raspberry Pi 2"""
    from tvm.tir.schedule import Schedule

    print("applying parallelization schedules...")
    optimized_funcs = {}

    for gv, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            try:
                func_name = gv.name_hint
                temp_mod = tvm.IRModule({func_name: func})
                sch = Schedule(temp_mod)

                # Conv2D optimization
                if ("conv" in func_name or "expand" in func_name or "project" in func_name or "last" in func_name) and "dw" not in func_name:
                    block = sch.get_block("conv", func_name=func_name)
                    loops = sch.get_loops(block)
                    if len(loops) == 7:
                        n, h, w, co, ci, kh, kw = loops
                        sch.parallel(h)

                # Depthwise Conv2D optimization
                elif "dw" in func_name and "relu" not in func_name:
                    block = sch.get_block("depthwise", func_name=func_name)
                    loops = sch.get_loops(block)
                    if len(loops) == 6:
                        n, h, w, c, kh, kw = loops
                        sch.parallel(h)

                # ReLU6 optimization
                elif "relu" in func_name:
                    block = sch.get_block("relu6", func_name=func_name)
                    loops = sch.get_loops(block)
                    if len(loops) == 4:
                        n, h, w, c = loops
                        sch.parallel(h)

                # Add optimization
                elif "add" in func_name:
                    block = sch.get_block("add", func_name=func_name)
                    loops = sch.get_loops(block)
                    if len(loops) == 4:
                        n, h, w, c = loops
                        sch.parallel(h)

                # Global pool optimization
                elif "pool" in func_name:
                    block = sch.get_block("global_pool", func_name=func_name)
                    loops = sch.get_loops(block)
                    if len(loops) == 4:
                        n, c, h, w = loops
                        sch.parallel(c)

                # Linear/Classifier optimization
                elif "classifier" in func_name or "linear" in func_name:
                    block = sch.get_block("linear", func_name=func_name)
                    loops = sch.get_loops(block)
                    if len(loops) == 3:
                        n, o, i = loops
                        sch.parallel(o)

                # Ensemble average
                elif "ensemble" in func_name:
                    block = sch.get_block("avg", func_name=func_name)
                    loops = sch.get_loops(block)
                    if len(loops) == 2:
                        n, c = loops
                        sch.parallel(c)

                optimized_funcs[gv.name_hint] = sch.mod[func_name]

            except Exception as e:
                optimized_funcs[gv.name_hint] = func

    opt_mod = tvm.IRModule()
    for name, func in optimized_funcs.items():
        opt_mod[name] = func

    print("  parallelization applied\n")
    return opt_mod


def build_sense_mobilenetv2_tir(onnx_path="../sense_onnx/model_main_17.onnx"):
    """Build complete TIR module for sense main model (m_main_1 + m_main_0 + ensemble)"""
    tir_funcs = {}

    print(f"Building m_main_1:")
    print()
    num_classes = sense_sub_model1(tir_funcs, onnx_path)

    print(f"Building m_main_0:")
    print()
    sense_sub_model0(tir_funcs, onnx_path, num_classes)

    ensemble(tir_funcs, num_classes)

    print()
    print(f"  generated {len(tir_funcs)} TIR functions")
    print()

    return tir_funcs


def build_tir_to_so(
    onnx_path="../sense_onnx/model_main_17.onnx",
    target_str=None,
    num_cores=4,
    save_path=None,
    opt_level=3
):
    """Build TIR functions and compile to shared library"""
    import os
    from tvm.ir.transform import PassContext

    if target_str is None:
        # target: c codegen
        target_str = f"c"

    if save_path is None:
        save_path = "../bin/sense_main_tir.tar"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Generate TIR functions
    tir_funcs = build_sense_mobilenetv2_tir(onnx_path)

    # Create IRModule
    mod = tvm.IRModule()
    for name, func in tir_funcs.items():
        mod[name] = func

    print()
    print(f"  created ir module with {len(mod.functions)} functions\n")

    # Apply parallelization schedules
    mod = schedule_op_for_rasp2(mod)

    target = tvm.target.Target(target_str)

    with PassContext(opt_level=opt_level):
        built_mod = tvm.build(mod, target=target)

    print(f"  build success\n")

    if save_path:
        built_mod.export_library(save_path)
        file_size = os.path.getsize(save_path) / 1024 / 1024
        print(f"  save module to: {save_path}")
        print(f"  size: {file_size:.2f} mb")

    return built_mod, mod


if __name__ == "__main__":
    # set params (relative to model directory)
    onnx_path = "../sense_onnx/model_main_17.onnx"
    save_path = "../bin/sense_main_tir.tar"
    weights_dir = "../bin/sense_params"
    num_cores = 4
    opt_level = 3

    if "--input" in sys.argv:
        onnx_path = sys.argv[sys.argv.index("--input") + 1]

    if "--output" in sys.argv:
        save_path = sys.argv[sys.argv.index("--output") + 1]

    if "--weights-dir" in sys.argv:
        weights_dir = sys.argv[sys.argv.index("--weights-dir") + 1]

    if "--cores" in sys.argv:
        num_cores = int(sys.argv[sys.argv.index("--cores") + 1])

    if "--opt-level" in sys.argv:
        opt_level = int(sys.argv[sys.argv.index("--opt-level") + 1])

    if "--target" in sys.argv:
        target_str = sys.argv[sys.argv.index("--target") + 1]
    else:
        target_str = f"llvm -mtriple=armv7l-linux-gnueabihf -mcpu=cortex-a7 -mattr=+neon -mfloat-abi=hard -num-cores={num_cores}"

    extract_weights = "--extract-weights" in sys.argv

    print("=" * 70)
    print(f"Configuration:")
    print(f"  Input: {onnx_path}")
    print(f"  Target: {target_str}")
    print(f"  Output: {save_path}")
    print(f"  Cores: {num_cores}")
    print(f"  Opt level: {opt_level}")
    print(f"  Extract weights: {extract_weights}")
    print()
    print("=" * 70)
    print()

    try:
        if extract_weights:
            from model.parser import extract_onnx_weights
            extract_onnx_weights(onnx_path, weights_dir)

        built_mod, mod = build_tir_to_so(
            onnx_path=onnx_path,
            target_str=target_str,
            num_cores=num_cores,
            save_path=save_path,
            opt_level=opt_level
        )

        print()
        print("=" * 70)
        print("Build completed successfully!")
        print("=" * 70)

    except Exception as e:
        import traceback
        print()
        print("=" * 70)
        print("Build failed!")
        print("=" * 70)
        traceback.print_exc()
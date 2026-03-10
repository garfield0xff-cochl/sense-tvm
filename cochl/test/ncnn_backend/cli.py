# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import tvm

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(REPO_ROOT))

from tvm.cochl.core.translate import translate_onnx  # noqa: E402
from tvm.cochl.framework.ncnn.kernel.weight_packer import (  # noqa: E402
    NcnnWeightPacker,
)
from tvm.cochl.framework.ncnn.kernel.op_packer import (  # noqa: E402
    infer_ncnn_function_name,
)

from cochl.test.ncnn_backend.helpers import emit_metadata, unmatched_reason, as_pattern_entry  # noqa: E402
from tvm.cochl.framework.ncnn.codegen.sources import NCNN_TO_STANDALONE  # noqa: E402
from cochl.test.ncnn_backend.match import symbol_for_entry  # noqa: E402
from cochl.test.ncnn_backend.libgen import (  # noqa: E402
    build_call_extern_module,
    collect_neon_sources,
    write_lib0_with_impl,
)
from cochl.test.ncnn_backend.wrappers import wrapper_source  # noqa: E402
from tvm.cochl.framework.ncnn.codegen.codegen import emit_main_entry_from_plan, align_entries_with_plan  # noqa: E402
from cochl.test.ncnn_backend.memory_plan import build_plan  # noqa: E402
from cochl.test.ncnn_backend.pipeline import build_ir_mod  # noqa: E402
from tvm.cochl.framework.ncnn.kernel.op_packer import build_pattern_entries  # noqa: E402
from tvm.cochl.framework.relax.standalone_packer import match_relax_const_idx  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    force_pack4_fallback = False
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("sense/onnx/model_main_17.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cochl/test"),
        help="Output directory for packed weights (lib/weights.bin) and metadata",
    )
    parser.add_argument(
        "--lib0",
        type=Path,
        default=Path("cochl/test/lib/lib0.c"),
        help="Path to output standalone lib0.c",
    )
    parser.add_argument(
        "--tvm-fallback",
        action="store_true",
        help="Use tvm_c operator fallback path (accuracy-first)",
    )
    args = parser.parse_args()

    # 1) tvm_c operator fallback path (accuracy-first)
    if args.tvm_fallback:
        from tvm.cochl.framework.tvm_c.codegen import codegen as tvm_codegen
        import importlib
        import tarfile
        from tvm import relax as tvm_relax

        ir_mod, input_info, output_info, weights, weight_order = translate_onnx(args.onnx)
        relax_mod = importlib.import_module("tvm.cochl.framework.tvm_c.relax.pass")
        relax_passes = relax_mod.get_sense_main_passes()
        if relax_passes:
            with tvm.transform.PassContext(opt_level=3):
                ir_mod = tvm.transform.Sequential(relax_passes)(ir_mod)

        tir_pipeline = relax_mod.get_unpacked_passes()
        compiled = tvm_relax.build(
            ir_mod,
            target=tvm.target.Target("c"),
            tir_pipeline=tir_pipeline,
        )

        lib_dir = args.output_dir / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        tar_path = lib_dir / "lib.tar"
        compiled.export_library(str(tar_path))
        with tarfile.open(tar_path, "r") as tar:
            member = tar.getmember("lib0.c") if "lib0.c" in tar.getnames() else None
            if member is None:
                print("  Error: lib0.c not found in export.")
                return 1
            member.name = "lib0.c"
            tar.extract(member, path=lib_dir)

        input_name = next(iter(input_info.keys()))
        output_name = next(iter(output_info.keys()))
        input_shape = input_info[input_name]["shape"]
        output_shape = output_info[output_name]["shape"]

        tvm_codegen.codegen(
            ir_mod,
            input_name,
            input_shape,
            output_shape,
            weights,
            weight_order,
            args.output_dir,
            model_name="main_entry",
            save_metadata=True,
        )

        print(f"  Standalone lib0.c: {lib_dir / 'lib0.c'}")
        print(f"  Main entry: {lib_dir / 'main_entry.c'}")
        return 0

    # 2) Parse ONNX + ncnn relax passes
    # run ncnn pattern passes before tvm_c lowering / mem passes
    ir_mod = build_ir_mod(args.onnx, apply_mem_passes=False)

    # 3) Build const index map (hash -> const idx, shape-agnostic)
    _, _, _, weights, weight_order = translate_onnx(args.onnx)
    const_idx_map: dict[str, int] = {}
    for idx, name in enumerate(weight_order):
        arr = weights[name]
        h = hashlib.sha1()
        h.update(arr.tobytes())
        h.update(str(arr.dtype).encode("utf-8"))
        const_idx_map[h.hexdigest()] = idx

    # 4) Build pattern table and operator map (with const indices)
    pattern_entries = build_pattern_entries(
        ir_mod,
        "rpi2",
        insert_pad=True,
        const_idx_map=const_idx_map,
    )
    matched_lines = []
    unmatched_lines = []
    matched_symbols: set[str] = set()
    for idx, entry in enumerate(pattern_entries):
        ncnn_name = infer_ncnn_function_name(as_pattern_entry(entry))
        symbol = symbol_for_entry(entry)
        line = f"{idx}: {entry['tvm_op']} -> {ncnn_name} params={json.dumps(entry['attrs'], sort_keys=True)}"
        if symbol is None:
            reason = unmatched_reason(entry, ncnn_name)
            unmatched_lines.append(f"{line} reason={reason}")
        else:
            matched_lines.append(line)
            matched_symbols.add(symbol)

    # Wrapper stubs are always emitted; ensure their targets are linked.
    matched_symbols.update(
        {
            "conv3x3s1_pack1to4_neon_standalone",
            "conv3x3s2_pack1to4_neon_standalone",
            "conv3x3s2_neon_standalone",
            "conv1x1s1_neon_standalone",
            "convdw3x3s1_pack4_neon_standalone",
            "convdw3x3s2_pack4_neon_standalone",
        }
    )

    # include packed convolution helpers if conv2d exists (used for pack4 conv1x1)
    if any(entry.get("tvm_op") == "relax.nn.conv2d" for entry in pattern_entries):
        matched_symbols.add("convolution_transform_kernel_packed_standalone")
        matched_symbols.add("convolution_packed_neon_standalone")

    emit_metadata(args.output_dir, matched_lines, unmatched_lines)

    # 5) Build tvm_c memory plan (for const index mapping)
    plan = build_plan(args.onnx)
    if any(op.func_name == "pad" for op in plan.operations):
        matched_symbols.add("pad2d_nchw")

    # 6) Pack weights for NCNN backend with pack4 transforms
    aligned, _ = align_entries_with_plan(pattern_entries, plan)

    # Build const idx -> slot map
    const_indices = set()
    for op in plan.operations:
        for arg in op.input_vars:
            if 'metadata["relax.expr.Constant"]' in arg:
                idx = match_relax_const_idx(arg)
                if idx is not None:
                    const_indices.add(idx)
    sorted_idx = sorted(const_indices)
    const_idx_to_slot = {const_idx: slot for slot, const_idx in enumerate(sorted_idx)}

    transforms: dict[int, np.ndarray] = {}

    def pack_conv3x3_pack1to4(weight: np.ndarray, out_c: int, in_c: int) -> np.ndarray:
        w = weight.reshape(out_c, in_c, 3, 3)
        outc4 = out_c // 4
        packed = np.zeros((outc4, in_c, 3, 3, 4), dtype=np.float32)
        for oc in range(out_c):
            oc4 = oc // 4
            lane = oc % 4
            packed[oc4, :, :, :, lane] = w[oc, :, :, :]
        return packed.reshape(-1)

    def pack_dw3x3_pack4(weight: np.ndarray, out_c: int) -> np.ndarray:
        if weight.ndim == 4:
            w = weight.reshape(out_c, -1, 3, 3)[:, 0, :, :]
        else:
            w = weight.reshape(out_c, 3, 3)
        outc4 = out_c // 4
        packed = np.zeros((outc4, 3, 3, 4), dtype=np.float32)
        for oc in range(out_c):
            oc4 = oc // 4
            lane = oc % 4
            packed[oc4, :, :, lane] = w[oc, :, :]
        return packed.reshape(-1)

    for entry, op in aligned:
        if entry.get("tvm_op") != "relax.nn.conv2d":
            continue
        ncnn_name = infer_ncnn_function_name(as_pattern_entry(entry))
        if len(op.input_vars) < 2:
            continue
        k_h = int(entry["attrs"].get("kernel_h", 0))
        k_w = int(entry["attrs"].get("kernel_w", 0))
        const_idx = match_relax_const_idx(op.input_vars[1])
        if const_idx is None:
            continue
        slot = const_idx_to_slot.get(const_idx)
        if slot is None:
            continue
        weight_name = weight_order[slot]
        weight = weights[weight_name].astype(np.float32)
        out_c = int(entry["attrs"].get("out_channels", weight.shape[0]))
        in_c = int(entry["attrs"].get("in_channels", weight.shape[1] if weight.ndim > 1 else 1))
        groups = int(entry["attrs"].get("groups", 1))
        if force_pack4_fallback:
            continue
        if ncnn_name in {"conv3x3s1_pack1to4_neon", "conv3x3s2_pack1to4_neon"}:
            if groups == 1 and k_h == 3 and k_w == 3 and weight.size == out_c * in_c * 9:
                transforms[slot] = pack_conv3x3_pack1to4(weight, out_c, in_c)
        elif ncnn_name in {"convdw3x3s1_pack4_neon", "convdw3x3s2_pack4_neon"}:
            if k_h == 3 and k_w == 3 and weight.size == out_c * 9:
                transforms[slot] = pack_dw3x3_pack4(weight, out_c)

    NcnnWeightPacker.pack(weights, weight_order, args.output_dir, save_metadata=True, transforms=transforms)

    # 5) Emit call_extern lib0.c via TIR build
    mod = build_call_extern_module(pattern_entries)
    rt_mod = tvm.build(mod, target="c")
    args.lib0.parent.mkdir(parents=True, exist_ok=True)
    rt_mod.write_to_file(str(args.lib0))

    # 6) Append wrappers + matched implementations
    sources = collect_neon_sources(matched_symbols)
    write_lib0_with_impl(args.lib0, sources, wrapper_source(), matched_symbols)

    # 7) Build tvm_c memory plan and emit main entry using storage buffers
    lib_dir = args.output_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    entry_path = lib_dir / "main_entry.c"
    if not getattr(plan, "scalar_values", None):
        try:
            import re
            from cochl.test.ncnn_backend import memory_plan
            ir_mod = memory_plan.build_tvmc_ir_mod(args.onnx)
            text = ir_mod.script(show_meta=True)
            m = re.search(r"T_multiply[^\n]*= .*?\* T\.float32\(([^)]+)\)", text)
            if m:
                plan.scalar_values = {"multiply": float(m.group(1))}
        except Exception:
            plan.scalar_values = {}
    weights_json = args.output_dir / "metadata" / "weights.json"
    emit_main_entry_from_plan(entry_path, plan, pattern_entries, weights_json, None)

    # 7) Export ncnn.h alongside artifacts
    ncnn_header = Path("cochl/include/target/nchw/ncnn.h")
    if ncnn_header.exists():
        shutil.copyfile(ncnn_header, lib_dir / "ncnn.h")

    print(f"  Standalone lib0.c: {args.lib0}")
    print(f"  Main entry: {entry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from tvm.cochl.framework.ncnn.codegen.sources import NCNN_TO_STANDALONE
from tvm.cochl.framework.ncnn.relax.op.pattern_table import infer_ncnn_function_name
from tvm.cochl.framework.ncnn.codegen.helpers import as_pattern_entry


def _binary_symbol(entry: dict, ncnn_func: str) -> str | None:
    tvm_op = entry.get("tvm_op", "").split(".")[-1]
    if ncnn_func == "binary_op_vector_broadcast_b":
        if tvm_op == "maximum":
            return "binary_op_vector_broadcast_b_max_standalone"
        if tvm_op == "minimum":
            return "binary_op_vector_broadcast_b_min_standalone"
        return "binary_op_vector_broadcast_b_add_standalone"
    if ncnn_func == "binary_op_vector_no_broadcast":
        if tvm_op == "maximum":
            return "binary_op_vector_no_broadcast_max_standalone"
        if tvm_op == "minimum":
            return "binary_op_vector_no_broadcast_min_standalone"
        return "binary_op_vector_no_broadcast_add_standalone"
    return None


def symbol_for_entry(entry: dict) -> str | None:
    ncnn_func = infer_ncnn_function_name(as_pattern_entry(entry))
    if ncnn_func.startswith("permute_order_"):
        return "permute_nd"
    if ncnn_func.startswith("binary_op_"):
        sym = _binary_symbol(entry, ncnn_func)
        if sym:
            return sym
    return NCNN_TO_STANDALONE.get(ncnn_func)


def call_extern_symbol(entry: dict) -> tuple[str, str] | None:
    ncnn_func = infer_ncnn_function_name(as_pattern_entry(entry))
    if ncnn_func == "conv3x3s1_pack1to4_neon":
        return "cochl_wrap_conv3x3s1_pack1to4", "conv"
    if ncnn_func == "conv3x3s2_pack1to4_neon":
        return "cochl_wrap_conv3x3s2_pack1to4", "conv"
    if ncnn_func == "conv3x3s2_neon":
        return "cochl_wrap_conv3x3s2_pack1", "conv"
    if ncnn_func == "conv1x1s1_neon":
        return "cochl_wrap_conv1x1s1", "conv"
    if ncnn_func == "convdw3x3s1_pack4_neon":
        return "cochl_wrap_convdw3x3s1", "conv"
    if ncnn_func == "convdw3x3s2_pack4_neon":
        return "cochl_wrap_convdw3x3s2", "conv"
    if ncnn_func == "binary_op_vector_broadcast_b":
        sym = _binary_symbol(entry, ncnn_func) or "binary_op_vector_broadcast_b_add_standalone"
        return sym, "binary"
    if ncnn_func == "binary_op_vector_no_broadcast":
        sym = _binary_symbol(entry, ncnn_func) or "binary_op_vector_no_broadcast_add_standalone"
        return sym, "binary"
    if ncnn_func == "binary_op_broadcast":
        return "binary_op_broadcast_add_standalone", "binary_broadcast"
    if ncnn_func == "sigmoid":
        return "sigmoid_neon_standalone", "sigmoid"
    if ncnn_func == "matmul_ta0_tb0":
        return "matmul_gemm_neon_standalone", "matmul"
    if ncnn_func == "Squeeze":
        return "squeeze_nd", "squeeze"
    if ncnn_func == "Permute" or ncnn_func.startswith("permute_order_"):
        return "permute_nd", "permute"
    if ncnn_func == "Padding":
        return "pad2d_nchw", "pad"
    if ncnn_func == "reduction_mean_2x3_k1":
        return "reduction_mean_hw_keepdims", "reduction_mean"
    return None

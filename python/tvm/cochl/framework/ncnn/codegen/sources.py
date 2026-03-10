# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[6]

NEON_SOURCES = {
    "conv3x3s2_neon_standalone": _REPO_ROOT / "cochl/src/target/neon/convolution_3x3.cc",
    "conv3x3s1_pack1to4_neon_standalone": _REPO_ROOT / "cochl/src/target/neon/convolution_3x3_pack1to4.cc",
    "conv3x3s2_pack1to4_neon_standalone": _REPO_ROOT / "cochl/src/target/neon/convolution_3x3_pack1to4.cc",
    "conv1x1s1_neon_standalone": _REPO_ROOT / "cochl/src/target/neon/convolution_1x1.cc",
    "convdw3x3s1_pack4_neon_standalone": _REPO_ROOT
    / "cochl/src/target/neon/convolutiondepthwise_3x3_pack4.cc",
    "convdw3x3s2_pack4_neon_standalone": _REPO_ROOT
    / "cochl/src/target/neon/convolutiondepthwise_3x3_pack4.cc",
    "pad2d_nchw": _REPO_ROOT / "cochl/src/target/neon/pad.cc",
    "binary_op_vector_broadcast_b_add_standalone": _REPO_ROOT / "cochl/src/target/neon/binaryop_arm.cc",
    "binary_op_vector_broadcast_b_max_standalone": _REPO_ROOT / "cochl/src/target/neon/binaryop_arm.cc",
    "binary_op_vector_broadcast_b_min_standalone": _REPO_ROOT / "cochl/src/target/neon/binaryop_arm.cc",
    "binary_op_vector_no_broadcast_add_standalone": _REPO_ROOT / "cochl/src/target/neon/binaryop_arm.cc",
    "binary_op_vector_no_broadcast_max_standalone": _REPO_ROOT / "cochl/src/target/neon/binaryop_arm.cc",
    "binary_op_vector_no_broadcast_min_standalone": _REPO_ROOT / "cochl/src/target/neon/binaryop_arm.cc",
    "binary_op_broadcast_add_standalone": _REPO_ROOT / "cochl/src/target/neon/binaryop_broadcast.cc",
    "sigmoid_neon_standalone": _REPO_ROOT / "cochl/src/target/neon/sigmoid_arm.cc",
    "matmul_gemm_neon_standalone": _REPO_ROOT / "cochl/src/target/neon/matmul_arm.cc",
    "squeeze_nd": _REPO_ROOT / "cochl/src/target/neon/squeeze.cc",
    "permute_nd": _REPO_ROOT / "cochl/src/target/neon/permute.cc",
    "reduction_mean_hw_keepdims": _REPO_ROOT / "cochl/src/target/neon/reduction_mean.cc",
    "convolution_transform_kernel_packed_standalone": _REPO_ROOT / "cochl/src/target/neon/convolution_packed.cc",
    "convolution_packed_neon_standalone": _REPO_ROOT / "cochl/src/target/neon/convolution_packed.cc",
}

NCNN_TO_STANDALONE = {
    "conv3x3s1_pack1to4_neon": "conv3x3s1_pack1to4_neon_standalone",
    "conv3x3s2_pack1to4_neon": "conv3x3s2_pack1to4_neon_standalone",
    "conv3x3s2_neon": "conv3x3s2_neon_standalone",
    "conv1x1s1_neon": "conv1x1s1_neon_standalone",
    "convdw3x3s1_pack4_neon": "convdw3x3s1_pack4_neon_standalone",
    "convdw3x3s2_pack4_neon": "convdw3x3s2_pack4_neon_standalone",
    "binary_op_vector_broadcast_b": "binary_op_vector_broadcast_b_add_standalone",
    "binary_op_vector_no_broadcast": "binary_op_vector_no_broadcast_add_standalone",
    "binary_op_broadcast": "binary_op_broadcast_add_standalone",
    "sigmoid": "sigmoid_neon_standalone",
    "matmul_ta0_tb0": "matmul_gemm_neon_standalone",
    "Squeeze": "squeeze_nd",
    "Permute": "permute_nd",
    "Padding": "pad2d_nchw",
    "reduction_mean_2x3_k1": "reduction_mean_hw_keepdims",
}

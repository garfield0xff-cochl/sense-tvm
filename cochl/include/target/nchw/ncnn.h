// SPDX-License-Identifier: Apache-2.0
#ifndef COCHL_INCLUDE_TARGET_NCHW_XNNPACK_H_
#define COCHL_INCLUDE_TARGET_NCHW_XNNPACK_H_

#include <stddef.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Global thread control (override in main if needed)
extern int cochl_ncnn_num_threads;

typedef struct {
    int w;
    int h;
    int c;
    int elempack;
    size_t cstep;
    float* data;
} MatMini;

int conv3x3s1_pack1to4_neon_standalone(const MatMini* bottom,
                                       MatMini* top,
                                       const MatMini* kernel,
                                       const MatMini* bias);

int conv3x3s2_pack1to4_neon_standalone(const MatMini* bottom,
                                       MatMini* top,
                                       const MatMini* kernel,
                                       const MatMini* bias);

int conv3x3s2_neon_standalone(const MatMini* bottom,
                              MatMini* top,
                              const MatMini* kernel,
                              const MatMini* bias);

int ncnn_binary_op_vector_broadcast_b_add(const float* ptr,
                                          const float* ptr1,
                                          float* outptr,
                                          int size,
                                          int elempack);

int binary_op_vector_broadcast_b_add_standalone(const float* ptr,
                                                const float* ptr1,
                                                float* outptr,
                                                int size,
                                                int elempack);

int ncnn_binary_op_vector_broadcast_b_max(const float* ptr,
                                          const float* ptr1,
                                          float* outptr,
                                          int size,
                                          int elempack);

int binary_op_vector_broadcast_b_max_standalone(const float* ptr,
                                                const float* ptr1,
                                                float* outptr,
                                                int size,
                                                int elempack);

int ncnn_binary_op_vector_broadcast_b_min(const float* ptr,
                                          const float* ptr1,
                                          float* outptr,
                                          int size,
                                          int elempack);

int binary_op_vector_broadcast_b_min_standalone(const float* ptr,
                                                const float* ptr1,
                                                float* outptr,
                                                int size,
                                                int elempack);

int ncnn_binary_op_vector_no_broadcast_add(const float* ptr,
                                           const float* ptr1,
                                           float* outptr,
                                           int size);

int binary_op_vector_no_broadcast_add_standalone(const float* ptr,
                                                 const float* ptr1,
                                                 float* outptr,
                                                 int size);

int ncnn_binary_op_vector_no_broadcast_max(const float* ptr,
                                           const float* ptr1,
                                           float* outptr,
                                           int size);

int binary_op_vector_no_broadcast_max_standalone(const float* ptr,
                                                 const float* ptr1,
                                                 float* outptr,
                                                 int size);

int ncnn_binary_op_vector_no_broadcast_min(const float* ptr,
                                           const float* ptr1,
                                           float* outptr,
                                           int size);

int binary_op_vector_no_broadcast_min_standalone(const float* ptr,
                                                 const float* ptr1,
                                                 float* outptr,
                                                 int size);

int binary_op_broadcast_add_standalone(const float* a,
                                       const float* b,
                                       float* out,
                                       int outer,
                                       int inner);

int conv1x1s1_neon_standalone(const MatMini* bottom,
                              MatMini* top,
                              const MatMini* kernel,
                              const MatMini* bias);

int sigmoid_neon_standalone(float* ptr, int size);

int matmul_gemm_neon_standalone(const float* A,
                                const float* B,
                                const float* bias,
                                float* C,
                                int M,
                                int K,
                                int N);

int squeeze_nd(const float* in,
               float* out,
               const int* in_shape,
               int in_dims,
               const int* axes,
               int axes_len);

int permute_nd(const float* in,
               float* out,
               const int* in_shape,
               int in_dims,
               const int* perm);

int pad2d_nchw(const float* input,
               float* output,
               int n,
               int c,
               int in_h,
               int in_w,
               int pad_top,
               int pad_left,
               int pad_bottom,
               int pad_right,
               float value);

int reduction_mean_hw_keepdims(const float* input,
                               float* output,
                               int n,
                               int c,
                               int h,
                               int w);

int convdw3x3s1_pack4_neon_standalone(const MatMini* bottom,
                                      MatMini* top,
                                      const MatMini* kernel,
                                      const MatMini* bias);

int convdw3x3s2_pack4_neon_standalone(const MatMini* bottom,
                                      MatMini* top,
                                      const MatMini* kernel,
                                      const MatMini* bias);

int convolution_transform_kernel_packed_standalone(const MatMini* kernel,
                                                    MatMini* kernel_tm,
                                                    int inch,
                                                    int outch,
                                                    int kernel_w,
                                                    int kernel_h);

int convolution_packed_neon_standalone(const MatMini* bottom,
                                       MatMini* top,
                                       const MatMini* kernel_tm,
                                       const MatMini* bias,
                                       int kernel_w,
                                       int kernel_h,
                                       int dilation_w,
                                       int dilation_h,
                                       int stride_w,
                                       int stride_h,
                                       int activation_type);

#ifdef __cplusplus
}
#endif

#endif  // COCHL_SRC_TARGET_NEON_CONV3X3S1_PACK1TO4_STANDALONE_H_

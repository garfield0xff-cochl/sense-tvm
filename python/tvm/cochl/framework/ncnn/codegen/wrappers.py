# SPDX-License-Identifier: Apache-2.0


def wrapper_source() -> str:
    return """// Generated wrappers for call_extern (safe no-op on null inputs)
#include \"cochl/include/target/nchw/ncnn.h\"

int cochl_wrap_conv3x3s1_pack1to4(const float* input,
                                  const float* weight,
                                  const float* bias,
                                  float* output,
                                  int in_c,
                                  int in_h,
                                  int in_w,
                                  int out_c,
                                  int out_h,
                                  int out_w)
{
    if (!input || !weight || !output) return -1;
    MatMini bottom = {in_w, in_h, in_c, 1, (size_t)in_h * in_w, (float*)input};
    int outc4 = out_c / 4;
    size_t kernel_cstep = (size_t)in_c * 9;
    MatMini top = {out_w, out_h, outc4, 4, (size_t)out_h * out_w * 4, output};
    MatMini kernel = {9 * in_c, 1, outc4, 4, kernel_cstep, (float*)weight};
    MatMini b = {0, 0, out_c / 4, 4, 0, (float*)bias};
    return conv3x3s1_pack1to4_neon_standalone(&bottom, &top, &kernel, bias ? &b : NULL);
}

int cochl_wrap_conv3x3s2_pack1to4(const float* input,
                                  const float* weight,
                                  const float* bias,
                                  float* output,
                                  int in_c,
                                  int in_h,
                                  int in_w,
                                  int out_c,
                                  int out_h,
                                  int out_w)
{
    if (!input || !weight || !output) return -1;
    MatMini bottom = {in_w, in_h, in_c, 1, (size_t)in_h * in_w, (float*)input};
    int outc4 = out_c / 4;
    size_t kernel_cstep = (size_t)in_c * 9;
    MatMini top = {out_w, out_h, outc4, 4, (size_t)out_h * out_w * 4, output};
    MatMini kernel = {9 * in_c, 1, outc4, 4, kernel_cstep, (float*)weight};
    MatMini b = {0, 0, out_c / 4, 4, 0, (float*)bias};
    return conv3x3s2_pack1to4_neon_standalone(&bottom, &top, &kernel, bias ? &b : NULL);
}

int cochl_wrap_conv3x3s2_pack1(const float* input,
                               const float* weight,
                               const float* bias,
                               float* output,
                               int in_c,
                               int in_h,
                               int in_w,
                               int out_c,
                               int out_h,
                               int out_w)
{
    if (!input || !weight || !output) return -1;
    MatMini bottom = {in_w, in_h, in_c, 1, (size_t)in_h * in_w, (float*)input};
    MatMini top = {out_w, out_h, out_c, 1, (size_t)out_h * out_w, output};
    MatMini kernel = {0, 0, out_c, 1, 0, (float*)weight};
    MatMini b = {0, 0, out_c, 1, 0, (float*)bias};
    return conv3x3s2_neon_standalone(&bottom, &top, &kernel, bias ? &b : NULL);
}

int cochl_wrap_conv1x1s1(const float* input,
                         const float* weight,
                         const float* bias,
                         float* output,
                         int in_c,
                         int in_h,
                         int in_w,
                         int out_c,
                         int out_h,
                         int out_w)
{
    if (!input || !weight || !output) return -1;
    MatMini bottom = {in_w, in_h, in_c, 1, (size_t)in_h * in_w, (float*)input};
    MatMini top = {out_w, out_h, out_c, 1, (size_t)out_h * out_w, output};
    MatMini kernel = {0, 0, out_c, 1, 0, (float*)weight};
    MatMini b = {0, 0, out_c, 1, 0, (float*)bias};
    return conv1x1s1_neon_standalone(&bottom, &top, &kernel, bias ? &b : NULL);
}

int cochl_wrap_convdw3x3s1(const float* input,
                           const float* weight,
                           const float* bias,
                           float* output,
                           int in_c,
                           int in_h,
                           int in_w,
                           int out_c,
                           int out_h,
                           int out_w)
{
    if (!input || !weight || !output) return -1;
    int outc4 = out_c / 4;
    MatMini bottom = {in_w, in_h, outc4, 4, (size_t)in_h * in_w * 4, (float*)input};
    MatMini top = {out_w, out_h, outc4, 4, (size_t)out_h * out_w * 4, output};
    MatMini kernel = {9, outc4, 1, 4, (size_t)9, (float*)weight};
    MatMini b = {0, 0, out_c / 4, 4, 0, (float*)bias};
    return convdw3x3s1_pack4_neon_standalone(&bottom, &top, &kernel, bias ? &b : NULL);
}

int cochl_wrap_convdw3x3s2(const float* input,
                           const float* weight,
                           const float* bias,
                           float* output,
                           int in_c,
                           int in_h,
                           int in_w,
                           int out_c,
                           int out_h,
                           int out_w)
{
    if (!input || !weight || !output) return -1;
    int outc4 = out_c / 4;
    MatMini bottom = {in_w, in_h, outc4, 4, (size_t)in_h * in_w * 4, (float*)input};
    MatMini top = {out_w, out_h, outc4, 4, (size_t)out_h * out_w * 4, output};
    MatMini kernel = {9, outc4, 1, 4, (size_t)9, (float*)weight};
    MatMini b = {0, 0, out_c / 4, 4, 0, (float*)bias};
    return convdw3x3s2_pack4_neon_standalone(&bottom, &top, &kernel, bias ? &b : NULL);
}
"""

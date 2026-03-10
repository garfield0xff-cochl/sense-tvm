// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "cochl/include/target/nchw/ncnn.h"

static inline void fill_row_f32(float* row, int width, float value)
{
#if __ARM_NEON
    float32x4_t v = vdupq_n_f32(value);
    int x = 0;
    for (; x + 3 < width; x += 4)
    {
        vst1q_f32(row + x, v);
    }
    for (; x < width; ++x)
    {
        row[x] = value;
    }
#else
    std::fill(row, row + width, value);
#endif
}

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
               float value)
{
    if (!input || !output) return -1;
    if (pad_top == 0 && pad_left == 0 && pad_bottom == 0 && pad_right == 0)
    {
        if (output != input)
        {
            std::memcpy(output, input, (size_t)n * c * in_h * in_w * sizeof(float));
        }
        return 0;
    }
    const int out_h = in_h + pad_top + pad_bottom;
    const int out_w = in_w + pad_left + pad_right;
    const size_t out_plane = (size_t)out_h * out_w;
    const size_t in_plane = (size_t)in_h * in_w;

    const size_t total_in = (size_t)n * c * in_h * in_w;
    const float* in_ptr = input;
    float* tmp = nullptr;
    const float* in_begin = input;
    const float* in_end = input + total_in;
    const float* out_begin = output;
    const float* out_end = output + (size_t)n * c * out_h * out_w;
    const bool overlap = !(out_end <= in_begin || out_begin >= in_end);
    if (overlap)
    {
        tmp = (float*)std::malloc(total_in * sizeof(float));
        if (!tmp) return -1;
        std::memcpy(tmp, input, total_in * sizeof(float));
        in_ptr = tmp;
    }

    for (int ni = 0; ni < n; ++ni)
    {
        for (int ci = 0; ci < c; ++ci)
        {
            float* outptr = output + ((size_t)ni * c + ci) * out_plane;
            const float* inptr = in_ptr + ((size_t)ni * c + ci) * in_plane;

            for (int y = 0; y < out_h; ++y)
            {
                float* row = outptr + (size_t)y * out_w;
                fill_row_f32(row, out_w, value);
            }

            for (int y = 0; y < in_h; ++y)
            {
                float* row = outptr + (size_t)(y + pad_top) * out_w + pad_left;
                std::memcpy(row, inptr + (size_t)y * in_w, in_w * sizeof(float));
            }
        }
    }
    if (tmp) std::free(tmp);
    return 0;
}

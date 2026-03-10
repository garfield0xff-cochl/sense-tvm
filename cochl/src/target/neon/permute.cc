// SPDX-License-Identifier: Apache-2.0
#include <stddef.h>

#include "cochl/include/target/nchw/ncnn.h"

int permute_nd(const float* in,
               float* out,
               const int* in_shape,
               int in_dims,
               const int* perm)
{
    if (!in || !out || !in_shape || !perm || in_dims <= 0) return -1;

    int out_shape[8];
    for (int i = 0; i < in_dims; ++i)
    {
        int p = perm[i];
        if (p < 0) p += in_dims;
        if (p < 0 || p >= in_dims) return -2;
        out_shape[i] = in_shape[p];
    }

    size_t in_stride[8];
    size_t out_stride[8];
    in_stride[in_dims - 1] = 1;
    out_stride[in_dims - 1] = 1;
    for (int i = in_dims - 2; i >= 0; --i)
    {
        in_stride[i] = in_stride[i + 1] * (size_t)in_shape[i + 1];
        out_stride[i] = out_stride[i + 1] * (size_t)out_shape[i + 1];
    }

    size_t total = 1;
    for (int i = 0; i < in_dims; ++i) total *= (size_t)out_shape[i];

    for (size_t idx = 0; idx < total; ++idx)
    {
        size_t tmp = idx;
        size_t in_index = 0;
        for (int i = 0; i < in_dims; ++i)
        {
            size_t coord = tmp / out_stride[i];
            tmp -= coord * out_stride[i];

            int p = perm[i];
            if (p < 0) p += in_dims;
            in_index += coord * in_stride[p];
        }
        out[idx] = in[in_index];
    }

    return 0;
}

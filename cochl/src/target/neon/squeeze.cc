// SPDX-License-Identifier: Apache-2.0
#include <cstring>

#include "cochl/include/target/nchw/ncnn.h"

int squeeze_nd(const float* in,
               float* out,
               const int* in_shape,
               int in_dims,
               const int* axes,
               int axes_len)
{
    if (!in || !out || !in_shape || in_dims <= 0) return -1;

    // Mark which axes to squeeze
    bool squeeze_axis[8] = {false};
    if (axes && axes_len > 0)
    {
        for (int i = 0; i < axes_len; ++i)
        {
            int ax = axes[i];
            if (ax < 0) ax += in_dims;
            if (ax < 0 || ax >= in_dims) return -2;
            squeeze_axis[ax] = true;
        }
    }
    else
    {
        for (int i = 0; i < in_dims; ++i)
            if (in_shape[i] == 1) squeeze_axis[i] = true;
    }

    // Validate squeeze dims are 1
    for (int i = 0; i < in_dims; ++i)
    {
        if (squeeze_axis[i] && in_shape[i] != 1) return -3;
    }

    // Squeeze is a reshape; data layout unchanged
    size_t total = 1;
    for (int i = 0; i < in_dims; ++i) total *= (size_t)in_shape[i];
    std::memcpy(out, in, total * sizeof(float));
    return 0;
}

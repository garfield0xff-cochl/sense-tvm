// SPDX-License-Identifier: Apache-2.0
#include "cochl/include/target/nchw/ncnn.h"

int reduction_mean_hw_keepdims(const float* input,
                               float* output,
                               int n,
                               int c,
                               int h,
                               int w)
{
    if (!input || !output) return -1;
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) return -2;

    const int hw = h * w;
    const float inv_hw = 1.f / (float)hw;

    for (int ni = 0; ni < n; ++ni)
    {
        for (int ci = 0; ci < c; ++ci)
        {
            const float* ptr = input + ((size_t)ni * c + ci) * hw;
            float sum = 0.f;
            for (int i = 0; i < hw; ++i)
            {
                sum += ptr[i];
            }
            output[(size_t)ni * c + ci] = sum * inv_hw;
        }
    }
    return 0;
}

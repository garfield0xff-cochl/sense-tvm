// SPDX-License-Identifier: Apache-2.0
#include "cochl/include/target/nchw/ncnn.h"

int binary_op_broadcast_add_standalone(const float* a,
                                       const float* b,
                                       float* out,
                                       int outer,
                                       int inner)
{
    if (!a || !b || !out) return -1;
    if (outer <= 0 || inner <= 0) return -2;

    for (int i = 0; i < outer; ++i)
    {
        const float* ap = a + (size_t)i * inner;
        float* op = out + (size_t)i * inner;
        for (int j = 0; j < inner; ++j)
        {
            op[j] = ap[j] + b[j];
        }
    }
    return 0;
}

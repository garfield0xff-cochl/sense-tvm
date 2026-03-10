// SPDX-License-Identifier: Apache-2.0
#include "cochl/include/target/nchw/ncnn.h"

int conv3x3s2_neon_standalone(const MatMini* bottom,
                              MatMini* top,
                              const MatMini* kernel,
                              const MatMini* bias)
{
    if (!bottom || !top || !kernel) return -1;

    const int w = bottom->w;
    const int h = bottom->h;
    const int inch = bottom->c;
    const int outw = top->w;
    const int outh = top->h;
    const int outch = top->c;

    const float* kernel_ptr = kernel->data;
    const float* bias_ptr = bias ? bias->data : nullptr;

    const int tailstep = w - 2 * outw + w;

    for (int p = 0; p < outch; ++p)
    {
        float* outptr = top->data + (size_t)p * top->cstep;
        const float* k0 = kernel_ptr + (size_t)p * inch * 9;
        const float bias0 = bias_ptr ? bias_ptr[p] : 0.f;

        for (int i = 0; i < outh; i++)
        {
            const float* r0 = bottom->data + (size_t)i * 2 * w;
            const float* r1 = r0 + w;
            const float* r2 = r1 + w;

            for (int j = 0; j < outw; j++)
            {
                float sum = bias0;
                const float* kptr = k0;
                const float* r0p = r0;
                const float* r1p = r1;
                const float* r2p = r2;

                for (int q = 0; q < inch; q++)
                {
                    sum += r0p[0] * kptr[0];
                    sum += r0p[1] * kptr[1];
                    sum += r0p[2] * kptr[2];
                    sum += r1p[0] * kptr[3];
                    sum += r1p[1] * kptr[4];
                    sum += r1p[2] * kptr[5];
                    sum += r2p[0] * kptr[6];
                    sum += r2p[1] * kptr[7];
                    sum += r2p[2] * kptr[8];

                    r0p += w * h;
                    r1p += w * h;
                    r2p += w * h;
                    kptr += 9;
                }

                outptr[j] = sum;
                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            outptr += outw;
            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }

    return 0;
}

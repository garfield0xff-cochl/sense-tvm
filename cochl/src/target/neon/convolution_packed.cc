// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstddef>
#include <vector>

#include "cochl/include/target/nchw/ncnn.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

struct Mat
{
    int w = 0;
    int h = 0;
    int c = 0;
    int elempack = 1;
    size_t cstep = 0;
    float* data = nullptr;

    Mat() = default;

    Mat(int _w, int _h, int _c, int _elempack, size_t _cstep, float* _data)
        : w(_w), h(_h), c(_c), elempack(_elempack), cstep(_cstep), data(_data)
    {}

    void create(int _w, int _h, int _c)
    {
        w = _w;
        h = _h;
        c = _c;
        elempack = 1;
        cstep = (size_t)w * h;
        // data pointer is managed by caller
    }

    Mat channel(int p) const
    {
        return Mat(w, h, 1, elempack, cstep, data + (size_t)p * cstep * elempack);
    }

    float* row(int y) const { return data + (size_t)y * w * elempack; }

    float operator[](int i) const { return data[i]; }

    void fill(float32x4_t v) const
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            for (int y = 0; y < h; ++y)
            {
                float* rowptr = data + (size_t)y * w * elempack;
                for (int x = 0; x < w; ++x)
                {
                    vst1q_f32(rowptr + x * elempack, v);
                }
            }
        }
        else
        {
            const float s = vgetq_lane_f32(v, 0);
            for (int y = 0; y < h; ++y)
            {
                float* rowptr = data + (size_t)y * w * elempack;
                for (int x = 0; x < w * elempack; ++x)
                {
                    rowptr[x] = s;
                }
            }
        }
#else
        (void)v;
#endif
    }

    void fill(float v) const
    {
#if __ARM_NEON
        fill(vdupq_n_f32(v));
#else
        for (int y = 0; y < h; ++y)
        {
            float* rowptr = data + (size_t)y * w * elempack;
            for (int x = 0; x < w * elempack; ++x)
            {
                rowptr[x] = v;
            }
        }
#endif
    }

    operator float*() const { return data; }
    operator const float*() const { return data; }
};

struct Option
{
    int num_threads = 1;
};

static inline float activation_ss(float v, int /*activation_type*/, const Mat& /*activation_params*/)
{
    return v;
}

#if __ARM_NEON
static inline float32x4_t activation_ps(float32x4_t v, int /*activation_type*/, const Mat& /*activation_params*/)
{
    return v;
}
#endif

// ---- copied from ncnn convolution_packed.h, adapted to local Mat/Option ----

static void convolution_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // kernel_tm is expected to be preallocated with the correct shape

    int q = 0;
#if __ARM_NEON
#if __aarch64__
    for (; q + 7 < outch; q += 8)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;
        const float* kptr4 = (const float*)kernel + (q + 4) * inch * maxk;
        const float* kptr5 = (const float*)kernel + (q + 5) * inch * maxk;
        const float* kptr6 = (const float*)kernel + (q + 6) * inch * maxk;
        const float* kptr7 = (const float*)kernel + (q + 7) * inch * maxk;

        float* g00 = kernel_tm.channel(q / 8);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;
                const float* k4 = kptr4 + p * maxk;
                const float* k5 = kptr5 + p * maxk;
                const float* k6 = kptr6 + p * maxk;
                const float* k7 = kptr7 + p * maxk;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;
                const float* k4 = kptr4 + p * maxk;
                const float* k5 = kptr5 + p * maxk;
                const float* k6 = kptr6 + p * maxk;
                const float* k7 = kptr7 + p * maxk;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;
                const float* k4 = kptr4 + p * maxk;
                const float* k5 = kptr5 + p * maxk;
                const float* k6 = kptr6 + p * maxk;
                const float* k7 = kptr7 + p * maxk;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    g00[4] = k4[k];
                    g00[5] = k5[k];
                    g00[6] = k6[k];
                    g00[7] = k7[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
            }
        }
        for (; p < inch; p++)
        {
            const float* k0 = kptr0 + p * maxk;
            const float* k1 = kptr1 + p * maxk;
            const float* k2 = kptr2 + p * maxk;
            const float* k3 = kptr3 + p * maxk;
            const float* k4 = kptr4 + p * maxk;
            const float* k5 = kptr5 + p * maxk;
            const float* k6 = kptr6 + p * maxk;
            const float* k7 = kptr7 + p * maxk;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00[2] = k2[k];
                g00[3] = k3[k];
                g00[4] = k4[k];
                g00[5] = k5[k];
                g00[6] = k6[k];
                g00[7] = k7[k];
                g00 += 8;
            }
        }
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;

#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = kernel_tm.channel(q / 4);
#endif

        int p = 0;
#if __aarch64__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }
        }
#endif // __aarch64__
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;
                const float* k2 = kptr2 + p * maxk;
                const float* k3 = kptr3 + p * maxk;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    g00[2] = k2[k];
                    g00[3] = k3[k];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }
        }
        for (; p < inch; p++)
        {
            const float* k0 = kptr0 + p * maxk;
            const float* k1 = kptr1 + p * maxk;
            const float* k2 = kptr2 + p * maxk;
            const float* k3 = kptr3 + p * maxk;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00[2] = k2[k];
                g00[3] = k3[k];
                g00 += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; q + 1 < outch; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;

#if __ARM_NEON
#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#endif
#else
        float* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __ARM_NEON
#if __aarch64__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
            }
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + p * maxk;
                const float* k1 = kptr1 + p * maxk;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[k];
                    g00[1] = k1[k];
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
            }
        }
        for (; p < inch; p++)
        {
            const float* k0 = kptr0 + p * maxk;
            const float* k1 = kptr1 + p * maxk;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k0[k];
                g00[1] = k1[k];
                g00 += 2;
            }
        }
    }
    for (; q < outch; q++)
    {
        const float* kptr = (const float*)kernel + q * inch * maxk;

#if __ARM_NEON
#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#endif
#else
        float* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __ARM_NEON
#if __aarch64__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + p * maxk;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[k];
                    k0 += maxk;
                    g00 += 1;
                }
            }
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + p * maxk;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[k];
                    k0 += maxk;
                    g00 += 1;
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + p * maxk;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[k];
                    k0 += maxk;
                    g00 += 1;
                }
            }
        }
        for (; p < inch; p++)
        {
            const float* k0 = kptr + p * maxk;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k0[k];
                g00++;
            }
        }
    }
}

// convolution_packed body is included from a separate file to keep this unit manageable
#include "cochl/src/target/neon/convolution_packed_body.inc"

extern "C" int convolution_transform_kernel_packed_standalone(const MatMini* kernel,
                                                               MatMini* kernel_tm,
                                                               int inch,
                                                               int outch,
                                                               int kernel_w,
                                                               int kernel_h)
{
    if (!kernel || !kernel_tm || !kernel->data || !kernel_tm->data)
        return -1;

    Mat k(kernel->w, kernel->h, kernel->c, kernel->elempack, kernel->cstep, kernel->data);
    Mat kt(kernel_tm->w, kernel_tm->h, kernel_tm->c, kernel_tm->elempack, kernel_tm->cstep, kernel_tm->data);

    convolution_transform_kernel_packed(k, kt, inch, outch, kernel_w, kernel_h);

    kernel_tm->w = kt.w;
    kernel_tm->h = kt.h;
    kernel_tm->c = kt.c;
    kernel_tm->elempack = kt.elempack;
    kernel_tm->cstep = kt.cstep;

    return 0;
}

extern "C" int convolution_packed_neon_standalone(const MatMini* bottom,
                                                   MatMini* top,
                                                   const MatMini* kernel_tm,
                                                   const MatMini* bias,
                                                   int kernel_w,
                                                   int kernel_h,
                                                   int dilation_w,
                                                   int dilation_h,
                                                   int stride_w,
                                                   int stride_h,
                                                   int activation_type)
{
    if (!bottom || !top || !kernel_tm || !bottom->data || !top->data || !kernel_tm->data)
        return -1;

    Mat bottom_m(bottom->w, bottom->h, bottom->c, bottom->elempack, bottom->cstep, bottom->data);
    Mat top_m(top->w, top->h, top->c, top->elempack, top->cstep, top->data);
    Mat kernel_tm_m(kernel_tm->w, kernel_tm->h, kernel_tm->c, kernel_tm->elempack, kernel_tm->cstep, kernel_tm->data);

    Mat bias_m;
    if (bias && bias->data)
        bias_m = Mat(bias->w, bias->h, bias->c, bias->elempack, bias->cstep, bias->data);

    Mat activation_params;
    Option opt;

    convolution_packed(bottom_m, top_m, kernel_tm_m, bias_m, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
    return 0;
}

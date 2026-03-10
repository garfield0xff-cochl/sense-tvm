// SPDX-License-Identifier: Apache-2.0
#include <math.h>

#include "cochl/include/target/nchw/ncnn.h"

#if __ARM_NEON
// Minimal exp_ps + sigmoid_ps extracted from ncnn neon_mathfun.h
#if defined(__aarch64__)
#define VFMAQ_F32(a, b, c) vfmaq_f32(a, b, c)
#else
#define VFMAQ_F32(a, b, c) vmlaq_f32(a, b, c)
#endif

static inline float32x4_t exp_ps(float32x4_t x)
{
    const float c_exp_hi = 88.3762626647949f;
    const float c_exp_lo = -88.3762626647949f;
    const float c_cephes_LOG2EF = 1.44269504088896341f;
    const float c_cephes_exp_C1 = 0.693359375f;
    const float c_cephes_exp_C2 = -2.12194440e-4f;
    const float c_cephes_exp_p0 = 1.9875691500E-4f;
    const float c_cephes_exp_p1 = 1.3981999507E-3f;
    const float c_cephes_exp_p2 = 8.3334519073E-3f;
    const float c_cephes_exp_p3 = 4.1665795894E-2f;
    const float c_cephes_exp_p4 = 1.6666665459E-1f;
    const float c_cephes_exp_p5 = 5.0000001201E-1f;

    float32x4_t one = vdupq_n_f32(1.f);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    float32x4_t fx = VFMAQ_F32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));
    float32x4_t tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));
    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(c_cephes_exp_p0);
    y = VFMAQ_F32(vdupq_n_f32(c_cephes_exp_p1), y, x);
    y = VFMAQ_F32(vdupq_n_f32(c_cephes_exp_p2), y, x);
    y = VFMAQ_F32(vdupq_n_f32(c_cephes_exp_p3), y, x);
    y = VFMAQ_F32(vdupq_n_f32(c_cephes_exp_p4), y, x);
    y = VFMAQ_F32(vdupq_n_f32(c_cephes_exp_p5), y, x);

    y = VFMAQ_F32(x, y, z);
    y = vaddq_f32(y, one);

    int32x4_t mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    return vmulq_f32(y, pow2n);
}

static inline float32x4_t sigmoid_ps(float32x4_t _v)
{
    float32x4_t _one = vdupq_n_f32(1.f);
    _v = vnegq_f32(_v);
    _v = exp_ps(_v);
    _v = vaddq_f32(_v, _one);
    float32x4_t _outp = vrecpeq_f32(_v);
    _outp = vmulq_f32(vrecpsq_f32(_v, _outp), _outp);
    return vmulq_f32(vrecpsq_f32(_v, _outp), _outp);
}
#endif

int sigmoid_neon_standalone(float* ptr, int size)
{
    int i = 0;
#if __ARM_NEON
#if __aarch64__
    for (; i + 15 < size; i += 16)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        float32x4_t _p2 = vld1q_f32(ptr + 8);
        float32x4_t _p3 = vld1q_f32(ptr + 12);
        _p0 = sigmoid_ps(_p0);
        _p1 = sigmoid_ps(_p1);
        _p2 = sigmoid_ps(_p2);
        _p3 = sigmoid_ps(_p3);
        vst1q_f32(ptr, _p0);
        vst1q_f32(ptr + 4, _p1);
        vst1q_f32(ptr + 8, _p2);
        vst1q_f32(ptr + 12, _p3);
        ptr += 16;
    }
#endif
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        _p0 = sigmoid_ps(_p0);
        _p1 = sigmoid_ps(_p1);
        vst1q_f32(ptr, _p0);
        vst1q_f32(ptr + 4, _p1);
        ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        _p = sigmoid_ps(_p);
        vst1q_f32(ptr, _p);
        ptr += 4;
    }
#endif
    for (; i < size; i++)
    {
        *ptr = 1.f / (1.f + expf(-*ptr));
        ptr++;
    }
    return 0;
}

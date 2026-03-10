// SPDX-License-Identifier: Apache-2.0
#include <stddef.h>

#include "cochl/include/target/nchw/ncnn.h"

struct binary_op_add
{
    float operator()(const float& x, const float& y) const { return x + y; }
#if __ARM_NEON
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vaddq_f32(x, y);
    }
#endif
};

struct binary_op_max
{
    float operator()(const float& x, const float& y) const { return x > y ? x : y; }
#if __ARM_NEON
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vmaxq_f32(x, y);
    }
#endif
};

struct binary_op_min
{
    float operator()(const float& x, const float& y) const { return x < y ? x : y; }
#if __ARM_NEON
    float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const
    {
        return vminq_f32(x, y);
    }
#endif
};

template<typename Op>
static void binary_op_vector_broadcast_b(const float* ptr,
                                         const float* ptr1,
                                         float* outptr,
                                         int size,
                                         int elempack)
{
    const Op op;
    const float b = *ptr1;

    int i = 0;
#if __ARM_NEON
    float32x4_t _b_128 = (elempack == 4) ? vld1q_f32(ptr1) : vdupq_n_f32(b);
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _outp = op(_p, _b_128);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif
    for (; i < size; i++)
    {
        *outptr = op(*ptr, b);
        ptr += 1;
        outptr += 1;
    }
}

template<typename Op>
static void binary_op_vector_no_broadcast(const float* ptr,
                                          const float* ptr1,
                                          float* outptr,
                                          int size)
{
    const Op op;

    int i = 0;
#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        float32x4_t _b = vld1q_f32(ptr1);
        float32x4_t _outp = op(_p, _b);
        vst1q_f32(outptr, _outp);
        ptr += 4;
        ptr1 += 4;
        outptr += 4;
    }
#endif
    for (; i < size; i++)
    {
        *outptr = op(*ptr, *ptr1);
        ptr += 1;
        ptr1 += 1;
        outptr += 1;
    }
}

int ncnn_binary_op_vector_broadcast_b_add(const float* ptr,
                                          const float* ptr1,
                                          float* outptr,
                                          int size,
                                          int elempack)
{
    binary_op_vector_broadcast_b<binary_op_add>(ptr, ptr1, outptr, size, elempack);
    return 0;
}

int binary_op_vector_broadcast_b_add_standalone(const float* ptr,
                                                const float* ptr1,
                                                float* outptr,
                                                int size,
                                                int elempack)
{
    return ncnn_binary_op_vector_broadcast_b_add(ptr, ptr1, outptr, size, elempack);
}

int ncnn_binary_op_vector_broadcast_b_max(const float* ptr,
                                          const float* ptr1,
                                          float* outptr,
                                          int size,
                                          int elempack)
{
    binary_op_vector_broadcast_b<binary_op_max>(ptr, ptr1, outptr, size, elempack);
    return 0;
}

int binary_op_vector_broadcast_b_max_standalone(const float* ptr,
                                                const float* ptr1,
                                                float* outptr,
                                                int size,
                                                int elempack)
{
    return ncnn_binary_op_vector_broadcast_b_max(ptr, ptr1, outptr, size, elempack);
}

int ncnn_binary_op_vector_broadcast_b_min(const float* ptr,
                                          const float* ptr1,
                                          float* outptr,
                                          int size,
                                          int elempack)
{
    binary_op_vector_broadcast_b<binary_op_min>(ptr, ptr1, outptr, size, elempack);
    return 0;
}

int binary_op_vector_broadcast_b_min_standalone(const float* ptr,
                                                const float* ptr1,
                                                float* outptr,
                                                int size,
                                                int elempack)
{
    return ncnn_binary_op_vector_broadcast_b_min(ptr, ptr1, outptr, size, elempack);
}

int ncnn_binary_op_vector_no_broadcast_add(const float* ptr,
                                           const float* ptr1,
                                           float* outptr,
                                           int size)
{
    binary_op_vector_no_broadcast<binary_op_add>(ptr, ptr1, outptr, size);
    return 0;
}

int binary_op_vector_no_broadcast_add_standalone(const float* ptr,
                                                 const float* ptr1,
                                                 float* outptr,
                                                 int size)
{
    return ncnn_binary_op_vector_no_broadcast_add(ptr, ptr1, outptr, size);
}

int ncnn_binary_op_vector_no_broadcast_max(const float* ptr,
                                           const float* ptr1,
                                           float* outptr,
                                           int size)
{
    binary_op_vector_no_broadcast<binary_op_max>(ptr, ptr1, outptr, size);
    return 0;
}

int binary_op_vector_no_broadcast_max_standalone(const float* ptr,
                                                 const float* ptr1,
                                                 float* outptr,
                                                 int size)
{
    return ncnn_binary_op_vector_no_broadcast_max(ptr, ptr1, outptr, size);
}

int ncnn_binary_op_vector_no_broadcast_min(const float* ptr,
                                           const float* ptr1,
                                           float* outptr,
                                           int size)
{
    binary_op_vector_no_broadcast<binary_op_min>(ptr, ptr1, outptr, size);
    return 0;
}

int binary_op_vector_no_broadcast_min_standalone(const float* ptr,
                                                 const float* ptr1,
                                                 float* outptr,
                                                 int size)
{
    return ncnn_binary_op_vector_no_broadcast_min(ptr, ptr1, outptr, size);
}

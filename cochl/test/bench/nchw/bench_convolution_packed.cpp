#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/option.h"
#include "3rdparty/ncnn/src/layer.h"
#include "3rdparty/ncnn/src/paramdict.h"
using namespace ncnn;

// minimal activation stubs for activation_type=0
static inline float activation_ss(float v, int /*activation_type*/, const ncnn::Mat& /*activation_params*/)
{
    return v;
}

#if __ARM_NEON
static inline float32x4_t activation_ps(float32x4_t v, int /*activation_type*/, const ncnn::Mat& /*activation_params*/)
{
    return v;
}
#endif

#include "3rdparty/ncnn/src/layer/arm/convolution_packed.h"

#include "cochl/include/target/nchw/ncnn.h"

// Provide a minimal Option ctor to avoid linking option.cpp
namespace ncnn {
Option::Option()
{
    std::memset(this, 0, sizeof(Option));
    num_threads = 1;
    lightmode = true;
    use_winograd_convolution = true;
    use_sgemm_convolution = true;
}
}  // namespace ncnn

static void fill_random(float* ptr, size_t total, float low = -1.f, float high = 1.f)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < total; ++i) ptr[i] = dist(rng);
}

static void compute_kernel_tm_dims(int inch, int outch, int kernel_w, int kernel_h, int& w, int& h, int& c)
{
    const int maxk = kernel_w * kernel_h;
#if __ARM_NEON
#if __aarch64__
    if (outch >= 8)
    {
        if (inch >= 8)
        {
            w = 8 * 8 * maxk;
            h = inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2;
            c = outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2;
            return;
        }
        if (inch >= 4)
        {
            w = 8 * 4 * maxk;
            h = inch / 4 + (inch % 4) / 2 + inch % 2;
            c = outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2;
            return;
        }
        if (inch >= 2)
        {
            w = 8 * 2 * maxk;
            h = inch / 2 + inch % 2;
            c = outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2;
            return;
        }
        w = 8 * maxk;
        h = inch;
        c = outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2;
        return;
    }
#endif
    if (outch >= 4)
    {
#if __aarch64__
        if (inch >= 8)
        {
            w = 4 * 8 * maxk;
            h = inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2;
            c = outch / 4 + (outch % 4) / 2 + outch % 2;
            return;
        }
#endif
        if (inch >= 4)
        {
            w = 4 * 4 * maxk;
            h = inch / 4 + (inch % 4) / 2 + inch % 2;
            c = outch / 4 + (outch % 4) / 2 + outch % 2;
            return;
        }
        if (inch >= 2)
        {
            w = 4 * 2 * maxk;
            h = inch / 2 + inch % 2;
            c = outch / 4 + (outch % 4) / 2 + outch % 2;
            return;
        }
        w = 4 * maxk;
        h = inch;
        c = outch / 4 + (outch % 4) / 2 + outch % 2;
        return;
    }
#endif
    if (outch >= 2)
    {
#if __ARM_NEON
#if __aarch64__
        if (inch >= 8)
        {
            w = 2 * 8 * maxk;
            h = inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2;
            c = outch / 2 + outch % 2;
            return;
        }
#endif
        if (inch >= 4)
        {
            w = 2 * 4 * maxk;
            h = inch / 4 + (inch % 4) / 2 + inch % 2;
            c = outch / 2 + outch % 2;
            return;
        }
#endif
        if (inch >= 2)
        {
            w = 2 * 2 * maxk;
            h = inch / 2 + inch % 2;
            c = outch / 2 + outch % 2;
            return;
        }
        w = 2 * maxk;
        h = inch;
        c = outch / 2 + outch % 2;
        return;
    }

#if __ARM_NEON
#if __aarch64__
    if (inch >= 8)
    {
        w = 8 * maxk;
        h = inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2;
        c = outch;
        return;
    }
#endif
    if (inch >= 4)
    {
        w = 4 * maxk;
        h = inch / 4 + (inch % 4) / 2 + inch % 2;
        c = outch;
        return;
    }
#endif
    if (inch >= 2)
    {
        w = 2 * maxk;
        h = inch / 2 + inch % 2;
        c = outch;
        return;
    }

    w = maxk;
    h = inch;
    c = outch;
}

int main()
{
    const int inw = 16;
    const int inh = 16;
    const int kernel_w = 3;
    const int kernel_h = 3;
    const int stride_w = 1;
    const int stride_h = 1;
    const int dilation_w = 1;
    const int dilation_h = 1;
    const int inch = 8;
    const int outch = 12;
    const bool use_pack4 = true;

    const int outw = (inw - (kernel_w - 1) * dilation_w - 1) / stride_w + 1;
    const int outh = (inh - (kernel_h - 1) * dilation_h - 1) / stride_h + 1;

    ncnn::Option opt;
    opt.num_threads = 1;

    size_t bottom_elems = (size_t)inw * inh * inch;
    size_t top_elems = (size_t)outw * outh * outch;
    size_t kernel_elems = (size_t)outch * inch * kernel_w * kernel_h;
    size_t bias_elems = (size_t)outch;

    float* bottom_buf = (float*)aligned_alloc(16, bottom_elems * sizeof(float));
    float* top_buf = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* top_buf_standalone = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* kernel_buf = (float*)aligned_alloc(16, kernel_elems * sizeof(float));
    float* bias_buf = (float*)aligned_alloc(16, bias_elems * sizeof(float));

    fill_random(bottom_buf, bottom_elems);
    fill_random(kernel_buf, kernel_elems);
    fill_random(bias_buf, bias_elems);

    ncnn::Mat bottom;
    ncnn::Mat top;

    if (use_pack4)
    {
        bottom = ncnn::Mat(inw, inh, inch / 4, bottom_buf, 16u, 4, 0);
        top = ncnn::Mat(outw, outh, outch / 4, top_buf, 16u, 4, 0);
    }
    else
    {
        bottom = ncnn::Mat(inw, inh, inch, bottom_buf, 4u, 0);
        top = ncnn::Mat(outw, outh, outch, top_buf, 4u, 0);
    }

    ncnn::Mat kernel(kernel_w * kernel_h, inch, outch, kernel_buf, 4u, 0);
    ncnn::Mat bias(outch, 1, 1, bias_buf, 4u, 0);

    int ktw = 0, kth = 0, ktc = 0;
    compute_kernel_tm_dims(inch, outch, kernel_w, kernel_h, ktw, kth, ktc);
    size_t kernel_tm_elems = (size_t)ktw * kth * ktc;
    float* kernel_tm_buf = (float*)aligned_alloc(16, kernel_tm_elems * sizeof(float));
    std::memset(kernel_tm_buf, 0, kernel_tm_elems * sizeof(float));

    MatMini kernel_tm_s;
    kernel_tm_s.w = ktw;
    kernel_tm_s.h = kth;
    kernel_tm_s.c = ktc;
    kernel_tm_s.elempack = 1;
    kernel_tm_s.cstep = (size_t)ktw * kth;
    kernel_tm_s.data = kernel_tm_buf;

    MatMini kernel_s;
    kernel_s.w = kernel_w * kernel_h;
    kernel_s.h = inch;
    kernel_s.c = outch;
    kernel_s.elempack = 1;
    kernel_s.cstep = (size_t)kernel_s.w * kernel_s.h;
    kernel_s.data = kernel_buf;

    convolution_transform_kernel_packed_standalone(&kernel_s, &kernel_tm_s, inch, outch, kernel_w, kernel_h);
    ncnn::Mat kernel_tm(ktw, kth, ktc, kernel_tm_buf, 4u, 0);

    // warmup
    convolution_packed(bottom, top, kernel_tm, bias, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, 0, ncnn::Mat(), opt);

    MatMini bottom_s;
    MatMini top_s;
    bottom_s.w = bottom.w;
    bottom_s.h = bottom.h;
    bottom_s.c = bottom.c;
    bottom_s.elempack = bottom.elempack;
    bottom_s.cstep = (size_t)bottom.w * bottom.h;
    bottom_s.data = bottom_buf;

    top_s.w = top.w;
    top_s.h = top.h;
    top_s.c = top.c;
    top_s.elempack = top.elempack;
    top_s.cstep = (size_t)top.w * top.h;
    top_s.data = top_buf_standalone;

    // kernel_tm_s already set above

    MatMini bias_s;
    bias_s.w = bias.w;
    bias_s.h = bias.h;
    bias_s.c = bias.c;
    bias_s.elempack = bias.elempack;
    bias_s.cstep = (size_t)bias.w * bias.h * bias.elempack;
    bias_s.data = bias_buf;

    convolution_packed_neon_standalone(&bottom_s, &top_s, &kernel_tm_s, &bias_s, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, 0);

    const int repeats = 50;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        convolution_packed(bottom, top, kernel_tm, bias, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, 0, ncnn::Mat(), opt);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double ncnn_mean = total_ms / repeats;
    double ncnn_min = min_ms;

    total_ms = 0.0;
    min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        convolution_packed_neon_standalone(&bottom_s, &top_s, &kernel_tm_s, &bias_s, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    std::printf("convolution_packed benchmark (pack4=%s)\n", use_pack4 ? "true" : "false");
    std::printf("  in:  N=1 C=%d H=%d W=%d\n", inch, inh, inw);
    std::printf("  out: N=1 C=%d H=%d W=%d\n", outch, outh, outw);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);

    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < top_elems; ++i)
    {
        double diff = std::abs(top_buf[i] - top_buf_standalone[i]);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
    }
    std::printf("  max abs diff: %.6e\n", max_abs);
    std::printf("  mean abs diff: %.6e\n", sum_abs / (double)top_elems);

    std::free(bottom_buf);
    std::free(top_buf);
    std::free(top_buf_standalone);
    std::free(kernel_buf);
    std::free(kernel_tm_buf);
    std::free(bias_buf);
    return 0;
}

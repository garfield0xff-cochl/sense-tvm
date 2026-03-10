// SPDX-License-Identifier: Apache-2.0
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/option.h"
using namespace ncnn;
#include "3rdparty/ncnn/src/layer/arm/convolution_3x3.h"
#include "cochl/include/target/nchw/ncnn.h"

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

static float max_abs_diff(const float* a, const float* b, size_t n)
{
    float maxv = 0.f;
    for (size_t i = 0; i < n; ++i)
    {
        float d = std::fabs(a[i] - b[i]);
        if (d > maxv) maxv = d;
    }
    return maxv;
}

int main()
{
    int inch = 32;
    int outch = 64;
    int outw = 28;
    int outh = 28;
    int inw = outw * 2 + 1;
    int inh = outh * 2 + 1;

    ncnn::Option opt;
    opt.num_threads = 1;

    size_t bottom_elems = (size_t)inw * inh * inch;
    size_t top_elems = (size_t)outw * outh * outch;
    size_t kernel_elems = (size_t)outch * inch * 9;
    size_t bias_elems = (size_t)outch;

    float* bottom_buf = (float*)aligned_alloc(16, bottom_elems * sizeof(float));
    float* top_buf = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* top_buf_standalone = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* kernel_buf = (float*)aligned_alloc(16, kernel_elems * sizeof(float));
    float* bias_buf = (float*)aligned_alloc(16, bias_elems * sizeof(float));

    ncnn::Mat bottom(inw, inh, inch, bottom_buf, 4u, 0);
    ncnn::Mat top(outw, outh, outch, top_buf, 4u, 0);
    ncnn::Mat kernel(inch * 9, 1, outch, kernel_buf, 4u, 0);
    ncnn::Mat bias(outch, 1, 1, bias_buf, 4u, 0);

    fill_random(bottom_buf, bottom_elems);
    fill_random(kernel_buf, kernel_elems);
    fill_random(bias_buf, bias_elems);

    MatMini bottom_s{inw, inh, inch, 1, (size_t)inw * inh, bottom_buf};
    MatMini top_s{outw, outh, outch, 1, (size_t)outw * outh, top_buf_standalone};
    MatMini kernel_s{inch * 9, 1, outch, 1, (size_t)inch * 9, kernel_buf};
    MatMini bias_s{outch, 1, 1, 1, (size_t)outch, bias_buf};

    conv3x3s2_neon(bottom, top, kernel, bias, opt);
    conv3x3s2_neon_standalone(&bottom_s, &top_s, &kernel_s, &bias_s);

    float diff = max_abs_diff(top_buf, top_buf_standalone, top_elems);
    std::printf("conv3x3s2_neon accuracy max abs diff: %.6e\n", diff);

    const int repeats = 50;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        conv3x3s2_neon(bottom, top, kernel, bias, opt);
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
        conv3x3s2_neon_standalone(&bottom_s, &top_s, &kernel_s, &bias_s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    std::printf("conv3x3s2_neon benchmark\n");
    std::printf("  in:  N=1 C=%d H=%d W=%d\n", inch, inh, inw);
    std::printf("  out: N=1 C=%d H=%d W=%d\n", outch, outh, outw);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);

    std::free(bottom_buf);
    std::free(top_buf);
    std::free(top_buf_standalone);
    std::free(kernel_buf);
    std::free(bias_buf);
    return 0;
}

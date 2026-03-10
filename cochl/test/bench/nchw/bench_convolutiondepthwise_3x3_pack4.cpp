// SPDX-License-Identifier: Apache-2.0
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>
#include <cstring>

#include "cochl/include/target/nchw/ncnn.h"

#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/option.h"
using namespace ncnn;
#include "3rdparty/ncnn/src/layer/arm/convolutiondepthwise_3x3_pack4.h"

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

int main()
{
    int channels = 64;  // must be multiple of 4
    int w = 56;
    int h = 56;
    int outw = w - 2;
    int outh = h - 2;
    int cpack = channels / 4;

    ncnn::Option opt;
    opt.num_threads = 1;

    size_t bottom_elems = (size_t)w * h * cpack * 4;
    size_t top_elems = (size_t)outw * outh * cpack * 4;
    size_t kernel_elems = (size_t)9 * cpack * 4;
    size_t bias_elems = (size_t)cpack * 4;

    float* bottom_buf = (float*)aligned_alloc(16, bottom_elems * sizeof(float));
    float* top_buf = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* top_buf_standalone = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* kernel_buf = (float*)aligned_alloc(16, kernel_elems * sizeof(float));
    float* bias_buf = (float*)aligned_alloc(16, bias_elems * sizeof(float));

    ncnn::Mat bottom(w, h, cpack, bottom_buf, 16u, 4, 0);
    ncnn::Mat top(outw, outh, cpack, top_buf, 16u, 4, 0);
    ncnn::Mat kernel(9, 1, cpack, kernel_buf, 16u, 4, 0);
    ncnn::Mat bias(cpack, 1, 1, bias_buf, 16u, 4, 0);

    fill_random(bottom_buf, bottom_elems);
    fill_random(kernel_buf, kernel_elems);
    fill_random(bias_buf, bias_elems);

    MatMini bottom_s{w, h, cpack, 4, (size_t)w * h * 4, bottom_buf};
    MatMini top_s{outw, outh, cpack, 4, (size_t)outw * outh * 4, top_buf_standalone};
    MatMini kernel_s{9, 1, cpack, 4, (size_t)9, kernel_buf};
    MatMini bias_s{cpack, 1, 1, 4, 0, bias_buf};

    convdw3x3s1_pack4_neon(bottom, top, kernel, bias, opt);
    convdw3x3s1_pack4_neon_standalone(&bottom_s, &top_s, &kernel_s, &bias_s);

    const int repeats = 100;
    double total_ms = 0.0;
    double min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        convdw3x3s1_pack4_neon(bottom, top, kernel, bias, opt);
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
        convdw3x3s1_pack4_neon_standalone(&bottom_s, &top_s, &kernel_s, &bias_s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    // correctness check
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < top_elems; ++i)
    {
        double diff = std::fabs((double)top_buf[i] - (double)top_buf_standalone[i]);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
    }
    double mean_abs = sum_abs / (double)top_elems;

    std::printf("convdw3x3s1_pack4_neon benchmark\n");
    std::printf("  in:  N=1 C=%d H=%d W=%d\n", channels, h, w);
    std::printf("  out: N=1 C=%d H=%d W=%d\n", channels, outh, outw);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);
    std::printf("  max abs diff: %.6e\n", max_abs);
    std::printf("  mean abs diff: %.6e\n", mean_abs);

    int outw2 = (w - 3) / 2 + 1;
    int outh2 = (h - 3) / 2 + 1;
    size_t top2_elems = (size_t)outw2 * outh2 * cpack * 4;
    float* top2_buf = (float*)aligned_alloc(16, top2_elems * sizeof(float));
    float* top2_buf_standalone = (float*)aligned_alloc(16, top2_elems * sizeof(float));

    ncnn::Mat top2(outw2, outh2, cpack, top2_buf, 16u, 4, 0);
    MatMini top2_s{outw2, outh2, cpack, 4, (size_t)outw2 * outh2 * 4, top2_buf_standalone};

    convdw3x3s2_pack4_neon(bottom, top2, kernel, bias, opt);
    convdw3x3s2_pack4_neon_standalone(&bottom_s, &top2_s, &kernel_s, &bias_s);

    total_ms = 0.0;
    min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        convdw3x3s2_pack4_neon(bottom, top2, kernel, bias, opt);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double ncnn_mean2 = total_ms / repeats;
    double ncnn_min2 = min_ms;

    total_ms = 0.0;
    min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        convdw3x3s2_pack4_neon_standalone(&bottom_s, &top2_s, &kernel_s, &bias_s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean2 = total_ms / repeats;
    double standalone_min2 = min_ms;

    max_abs = 0.0;
    sum_abs = 0.0;
    for (size_t i = 0; i < top2_elems; ++i)
    {
        double diff = std::fabs((double)top2_buf[i] - (double)top2_buf_standalone[i]);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
    }
    mean_abs = sum_abs / (double)top2_elems;

    std::printf("convdw3x3s2_pack4_neon benchmark\n");
    std::printf("  in:  N=1 C=%d H=%d W=%d\n", channels, h, w);
    std::printf("  out: N=1 C=%d H=%d W=%d\n", channels, outh2, outw2);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean2);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min2);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean2);
    std::printf("  standalone min:  %.3f ms\n", standalone_min2);
    std::printf("  max abs diff: %.6e\n", max_abs);
    std::printf("  mean abs diff: %.6e\n", mean_abs);

    std::free(bottom_buf);
    std::free(top_buf);
    std::free(top_buf_standalone);
    std::free(top2_buf);
    std::free(top2_buf_standalone);
    std::free(kernel_buf);
    std::free(bias_buf);
    return 0;
}

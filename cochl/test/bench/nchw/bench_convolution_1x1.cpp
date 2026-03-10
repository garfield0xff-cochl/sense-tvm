// SPDX-License-Identifier: Apache-2.0
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "cochl/include/target/nchw/ncnn.h"

#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/option.h"
using namespace ncnn;
#include "3rdparty/ncnn/src/layer/arm/convolution_1x1.h"

static void fill_random(float* ptr, size_t total, float low = -1.f, float high = 1.f)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < total; ++i) ptr[i] = dist(rng);
}

int main()
{
    int inch = 64;
    int outch = 128;
    int outw = 56;
    int outh = 56;
    int inw = outw;
    int inh = outh;
    const bool use_pack4 = true;

    ncnn::Option opt;
    opt.num_threads = 1;

    size_t bottom_elems = (size_t)inw * inh * inch;
    size_t top_elems = (size_t)outw * outh * outch;
    size_t kernel_elems = (size_t)inch * outch;
    size_t bias_elems = (size_t)outch;

    float* bottom_buf = (float*)aligned_alloc(16, bottom_elems * sizeof(float));
    float* top_buf = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* top_buf_standalone = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* kernel_buf = (float*)aligned_alloc(16, kernel_elems * sizeof(float));
    float* bias_buf = (float*)aligned_alloc(16, bias_elems * sizeof(float));

    ncnn::Mat bottom(inw, inh, inch, bottom_buf, 4u, 0);
    ncnn::Mat top(outw, outh, outch, top_buf, 4u, 1, 0);
    ncnn::Mat kernel(inch, 1, outch, kernel_buf, 4u, 1, 0);
    ncnn::Mat bias(outch, 1, 1, bias_buf, 4u, 1, 0);

    fill_random(bottom_buf, bottom_elems);
    fill_random(kernel_buf, kernel_elems);
    fill_random(bias_buf, bias_elems);

    MatMini bottom_s{inw, inh, inch, 1, (size_t)inw * inh, bottom_buf};
    MatMini top_s{outw, outh, outch, 1, (size_t)outw * outh, top_buf_standalone};
    MatMini kernel_s{inch, 1, outch, 1, (size_t)inch, kernel_buf};
    MatMini bias_s{outch, 1, 1, 1, (size_t)outch, bias_buf};

    if (use_pack4)
    {
        int outch_pack = outch / 4;
        top_elems = (size_t)outw * outh * outch_pack * 4;
        kernel_elems = (size_t)inch * outch_pack * 4;
        bias_elems = (size_t)outch_pack * 4;

        top = ncnn::Mat(outw, outh, outch_pack, top_buf, 16u, 4, 0);
        kernel = ncnn::Mat(inch, 1, outch_pack, kernel_buf, 16u, 4, 0);
        bias = ncnn::Mat(outch_pack, 1, 1, bias_buf, 16u, 4, 0);

        top_s = MatMini{outw, outh, outch_pack, 4, (size_t)outw * outh * 4, top_buf_standalone};
        kernel_s = MatMini{inch, 1, outch_pack, 4, (size_t)inch * 4, kernel_buf};
        bias_s = MatMini{outch_pack, 1, 1, 4, (size_t)outch_pack * 4, bias_buf};
    }

    // warmup
    conv1x1s1_neon(bottom, top, kernel, bias, opt);
    conv1x1s1_neon_standalone(&bottom_s, &top_s, &kernel_s, &bias_s);

    const int repeats = 100;
    double total_ms = 0.0;
    double min_ms = 1e30;

    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        conv1x1s1_neon(bottom, top, kernel, bias, opt);
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
        conv1x1s1_neon_standalone(&bottom_s, &top_s, &kernel_s, &bias_s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    std::printf("conv1x1s1_neon benchmark\n");
    std::printf("  in:  N=1 C=%d H=%d W=%d\n", inch, inh, inw);
    std::printf("  out: N=1 C=%d H=%d W=%d\n", outch, outh, outw);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);

    // correctness check (ncnn vs standalone)
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
    std::free(bias_buf);
    return 0;
}

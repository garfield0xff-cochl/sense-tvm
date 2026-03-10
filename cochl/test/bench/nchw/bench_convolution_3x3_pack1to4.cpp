#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "3rdparty/ncnn/src/mat.h"
#include "3rdparty/ncnn/src/option.h"
using namespace ncnn;
#include "3rdparty/ncnn/src/layer/arm/convolution_3x3_pack1to4.h"
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

static size_t align_cstep(size_t w, size_t h, size_t elemsize)
{
    size_t bytes = w * h * elemsize;
    size_t aligned = (bytes + 15) / 16 * 16;
    return aligned / elemsize;
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

int main(int argc, char** argv)
{
    int inch = 32;
    int outch = 64; // must be multiple of 4 for pack1to4
    int outw = 56;
    int outh = 56;
    int inw = outw + 2;
    int inh = outh + 2;

    int outch_pack = outch / 4;

    ncnn::Option opt;
    opt.num_threads = 1;

    // allocate raw buffers and use external-data Mat constructors (avoid Mat::create)
    size_t bottom_elems = (size_t)inw * inh * inch;
    size_t top_elems = (size_t)outw * outh * outch_pack * 4;
    size_t kernel_elems = (size_t)(inch * 9) * outch_pack * 4;
    size_t bias_elems = (size_t)outch_pack * 4;

    float* bottom_buf = (float*)aligned_alloc(16, bottom_elems * sizeof(float));
    float* top_buf = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* top_buf_standalone = (float*)aligned_alloc(16, top_elems * sizeof(float));
    float* kernel_buf = (float*)aligned_alloc(16, kernel_elems * sizeof(float));
    float* bias_buf = (float*)aligned_alloc(16, bias_elems * sizeof(float));

    ncnn::Mat bottom(inw, inh, inch, bottom_buf, 4u, 0);
    ncnn::Mat top(outw, outh, outch_pack, top_buf, 16u, 4, 0);
    ncnn::Mat kernel(inch * 9, 1, outch_pack, kernel_buf, 16u, 4, 0);
    ncnn::Mat bias(outch_pack, 1, 1, bias_buf, 16u, 4, 0);

    fill_random(bottom_buf, bottom_elems);
    fill_random(kernel_buf, kernel_elems);
    fill_random(bias_buf, bias_elems);

    // warmup (s1)
    conv3x3s1_pack1to4_neon(bottom, top, kernel, bias, opt);
    size_t bottom_cstep = align_cstep(inw, inh, 4);
    MatMini bottom_s{inw, inh, inch, 1, bottom_cstep, bottom_buf};
    MatMini top_s{outw, outh, outch_pack, 4, (size_t)outw * outh * 4, top_buf_standalone};
    MatMini kernel_s{inch * 9, 1, outch_pack, 4, (size_t)(inch * 9) * 4, kernel_buf};
    MatMini bias_s{outch_pack, 1, 1, 4, (size_t)outch_pack * 4, bias_buf};
    conv3x3s1_pack1to4_neon_standalone(&bottom_s, &top_s, &kernel_s, &bias_s);
    float diff_s1 = max_abs_diff(top_buf, top_buf_standalone, top_elems);

    const int repeats = 50;
    double total_ms = 0.0;
    double min_ms = 1e30;

    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        conv3x3s1_pack1to4_neon(bottom, top, kernel, bias, opt);
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
        conv3x3s1_pack1to4_neon_standalone(&bottom_s, &top_s, &kernel_s, &bias_s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }

    double standalone_mean = total_ms / repeats;
    double standalone_min = min_ms;

    std::printf("conv3x3s1_pack1to4_neon benchmark\n");
    std::printf("  in:  N=1 C=%d H=%d W=%d\n", inch, inh, inw);
    std::printf("  out: N=1 C=%d H=%d W=%d\n", outch, outh, outw);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean);
    std::printf("  standalone min:  %.3f ms\n", standalone_min);
    std::printf("  max abs diff (s1): %.6g\n", diff_s1);

    // stride=2 benchmark
    int outw2 = outw / 2;
    int outh2 = outh / 2;
    int inw2 = outw2 * 2 + 1;
    int inh2 = outh2 * 2 + 1;

    size_t bottom_elems2 = (size_t)inw2 * inh2 * inch;
    size_t top_elems2 = (size_t)outw2 * outh2 * outch_pack * 4;

    float* bottom_buf2 = (float*)aligned_alloc(16, bottom_elems2 * sizeof(float));
    float* top_buf2 = (float*)aligned_alloc(16, top_elems2 * sizeof(float));
    float* top_buf2_standalone = (float*)aligned_alloc(16, top_elems2 * sizeof(float));

    ncnn::Mat bottom2(inw2, inh2, inch, bottom_buf2, 4u, 0);
    ncnn::Mat top2(outw2, outh2, outch_pack, top_buf2, 16u, 4, 0);

    fill_random(bottom_buf2, bottom_elems2);

    size_t bottom2_cstep = align_cstep(inw2, inh2, 4);
    MatMini bottom2_s{inw2, inh2, inch, 1, bottom2_cstep, bottom_buf2};
    MatMini top2_s{outw2, outh2, outch_pack, 4, (size_t)outw2 * outh2 * 4, top_buf2_standalone};

    conv3x3s2_pack1to4_neon(bottom2, top2, kernel, bias, opt);
    conv3x3s2_pack1to4_neon_standalone(&bottom2_s, &top2_s, &kernel_s, &bias_s);
    float diff_s2 = max_abs_diff(top_buf2, top_buf2_standalone, top_elems2);

    total_ms = 0.0;
    min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        conv3x3s2_pack1to4_neon(bottom2, top2, kernel, bias, opt);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double ncnn_mean_s2 = total_ms / repeats;
    double ncnn_min_s2 = min_ms;

    total_ms = 0.0;
    min_ms = 1e30;
    for (int i = 0; i < repeats; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        conv3x3s2_pack1to4_neon_standalone(&bottom2_s, &top2_s, &kernel_s, &bias_s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }
    double standalone_mean_s2 = total_ms / repeats;
    double standalone_min_s2 = min_ms;

    std::printf("conv3x3s2_pack1to4_neon benchmark\n");
    std::printf("  in:  N=1 C=%d H=%d W=%d\n", inch, inh2, inw2);
    std::printf("  out: N=1 C=%d H=%d W=%d\n", outch, outh2, outw2);
    std::printf("  ncnn mean: %.3f ms\n", ncnn_mean_s2);
    std::printf("  ncnn min:  %.3f ms\n", ncnn_min_s2);
    std::printf("  standalone mean: %.3f ms\n", standalone_mean_s2);
    std::printf("  standalone min:  %.3f ms\n", standalone_min_s2);
    std::printf("  max abs diff (s2): %.6g\n", diff_s2);

    std::free(bottom_buf);
    std::free(top_buf);
    std::free(top_buf_standalone);
    std::free(kernel_buf);
    std::free(bias_buf);
    std::free(bottom_buf2);
    std::free(top_buf2);
    std::free(top_buf2_standalone);
    return 0;
}

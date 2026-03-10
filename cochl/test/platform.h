#ifndef NCNN_PLATFORM_H
#define NCNN_PLATFORM_H

#if defined(_MSC_VER)
#define NCNN_FORCEINLINE __forceinline
#else
#define NCNN_FORCEINLINE inline __attribute__((always_inline))
#endif

#define NCNN_EXPORT
#define NCNN_THREADS 0
#define NCNN_OPENMP 0
#define NCNN_PLATFORM_API 0
#define NCNN_PIXEL 0
#define NCNN_PIXEL_ROTATE 0
#define NCNN_PIXEL_AFFINE 0
#define NCNN_VULKAN 0

#endif // NCNN_PLATFORM_H

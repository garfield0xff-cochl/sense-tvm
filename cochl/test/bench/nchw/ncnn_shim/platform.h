// Minimal platform.h for local bench builds (do not modify 3rdparty/ncnn/src)
#ifndef NCNN_PLATFORM_H
#define NCNN_PLATFORM_H

#define NCNN_STDIO 1
#define NCNN_STRING 1
#define NCNN_SIMPLESTL 1
#define NCNN_THREADS 0
#define NCNN_VULKAN 0
#define NCNN_BF16 0
#define NCNN_ARM82 0

#define NCNN_FORCEINLINE inline

#ifndef NCNN_EXPORT
#define NCNN_EXPORT
#endif

#include <stdio.h>
#include <stddef.h>
#include <string>
#include <vector>

#ifndef NCNN_LOGE
#define NCNN_LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif

#endif  // NCNN_PLATFORM_H

// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <string>
#include <vector>

#include "3rdparty/ncnn/src/layer.h"
#include "3rdparty/ncnn/src/option.h"

namespace ncnn {

Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = false;
    support_packing = false;
    support_bf16_storage = false;
    support_fp16_storage = false;
    support_int8_storage = false;
    support_tensor_storage = false;
    support_vulkan_packing = false;
    support_any_packing = false;
    support_vulkan_any_packing = false;
    support_reserved_1 = false;
    support_reserved_2 = false;
    support_reserved_3 = false;
    featmask = 0;
#if NCNN_VULKAN
    vkdev = 0;
#endif
    userdata = 0;
    typeindex = -1;
}

Layer::~Layer() {}

int Layer::load_param(const ParamDict&) { return 0; }
int Layer::load_model(const ModelBin&) { return 0; }
int Layer::create_pipeline(const Option&) { return 0; }
int Layer::destroy_pipeline(const Option&) { return 0; }
int Layer::forward(const std::vector<Mat>&, std::vector<Mat>&, const Option&) const { return -1; }
int Layer::forward(const Mat&, Mat&, const Option&) const { return -1; }
int Layer::forward_inplace(std::vector<Mat>&, const Option&) const { return -1; }
int Layer::forward_inplace(Mat&, const Option&) const { return -1; }

Option::Option()
{
    std::memset(this, 0, sizeof(Option));
    num_threads = 1;
    blob_allocator = 0;
    workspace_allocator = 0;
    use_winograd_convolution = true;
    use_sgemm_convolution = true;
}

Layer* create_layer(const char*)
{
    return 0;
}

Layer* create_layer(int)
{
    return 0;
}

}  // namespace ncnn

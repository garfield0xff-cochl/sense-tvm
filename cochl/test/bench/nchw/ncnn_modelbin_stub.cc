// SPDX-License-Identifier: Apache-2.0
#include "3rdparty/ncnn/src/modelbin.h"

namespace ncnn {

ModelBin::ModelBin() {}
ModelBin::~ModelBin() {}

Mat ModelBin::load(int, int) const { return Mat(); }
Mat ModelBin::load(int, int, int) const { return Mat(); }
Mat ModelBin::load(int, int, int, int) const { return Mat(); }
Mat ModelBin::load(int, int, int, int, int) const { return Mat(); }

ModelBinFromMatArray::ModelBinFromMatArray(const Mat*)
    : d(0)
{
}

ModelBinFromMatArray::~ModelBinFromMatArray()
{
}

Mat ModelBinFromMatArray::load(int, int) const
{
    return Mat();
}

}  // namespace ncnn

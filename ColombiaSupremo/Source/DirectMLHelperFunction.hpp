// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <winrt/base.h>

#include "Tensor.hpp"
#include "d3dx12.h"

namespace colombia_supremo::dml_helper {

winrt::com_ptr<IDMLDevice> CreateDevice(winrt::com_ptr<ID3D12Device> d3D12Device);

winrt::com_ptr<IDMLCommandRecorder>
    CreateCommandRecorder(winrt::com_ptr<IDMLDevice> device);

uint64_t CalculateBufferTensorSize(const TensorShape &tensorShape);

TensorShape CalculateStrides(const TensorShape &tensorShape);

} // namespace colombia_supremo::dml_helper

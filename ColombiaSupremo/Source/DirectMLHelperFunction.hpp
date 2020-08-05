// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "Tensor.hpp"

namespace colombia_supremo::dml_helper {

uint64_t CalculateBufferTensorSize(const TensorShape &tensorShape);

TensorShape CalculateStrides(const TensorShape &tensorShape);

} // namespace colombia_supremo::dml_helper

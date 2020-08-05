// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "DirectMLHelperFunction.hpp"
#include <cassert>

namespace colombia_supremo::dml_helper {

uint64_t CalculateBufferTensorSize(const TensorShape &tensorShape) {
  uint64_t tensorElementCount = 1;

  for (const auto &element : tensorShape) {
    tensorElementCount *= static_cast<uint64_t>(element);
  }

  const uint64_t elementSizeInBytes = 4; // we support only 32-bit float

  return tensorElementCount * elementSizeInBytes;
}

TensorShape CalculateStrides(const TensorShape &tensorShape) {
  TensorShape stride;
  stride.reserve(tensorShape.size());

  // only NCHW allowed
  stride.emplace_back(tensorShape[1] * tensorShape[2] * tensorShape[3]);
  stride.emplace_back(tensorShape[2] * tensorShape[3]);
  stride.emplace_back(tensorShape[3]);
  stride.emplace_back(1);

  return stride;
}

} // namespace colombia_supremo::dml_helper

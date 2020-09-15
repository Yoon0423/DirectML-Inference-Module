// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "InferenceEngine.hpp"

#include <vector>

namespace colombia_supremo {

class MultiSessionEngine: public InferenceEngine {
public:
  MultiSessionEngine() = delete;
  MultiSessionEngine(const char *const modelFilePath, const size_t batchSize);
  ~MultiSessionEngine() = default;

  void Run(const std::vector<TensorRawData> &inputs,
           std::vector<TensorRawData> &outputs) override;
};

} // namespace colombia_supremo

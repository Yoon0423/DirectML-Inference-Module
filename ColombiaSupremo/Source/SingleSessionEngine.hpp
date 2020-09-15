// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "InferenceEngine.hpp"

#include <vector>

namespace colombia_supremo {

class SingleSessionEngine : public InferenceEngine {
public:
  SingleSessionEngine() = delete;
  SingleSessionEngine(const char *const modelFilePath);
  ~SingleSessionEngine() = default;

  void Run(const std::vector<TensorRawData> &inputs,
           std::vector<TensorRawData> &outputs) override;
};

} // namespace colombia_supremo

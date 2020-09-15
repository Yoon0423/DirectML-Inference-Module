// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Model.hpp"
#include "Session.hpp"

#include <vector>

namespace colombia_supremo {

class InferenceEngine {
public:
  InferenceEngine() = delete;
  InferenceEngine(const char *const modelFilePath, const size_t batchSize);
  ~InferenceEngine() = default;

  virtual void Run(const std::vector<TensorRawData> &inputs,
                   std::vector<TensorRawData> &outputs) = 0;

protected:
  const size_t mBatchSize;
  Model mModel;
  std::vector<Session> mSessions;
};

} // namespace colombia_supremo

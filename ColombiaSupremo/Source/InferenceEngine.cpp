// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "InferenceEngine.hpp"

#include <cassert>
#include <future>
#include <thread>

namespace colombia_supremo {

InferenceEngine::InferenceEngine(const char *const modelFilePath,
                                 const size_t batchSize)
    : mModel(Model(modelFilePath)), mBatchSize(batchSize) {
  mSessions.reserve(batchSize);

  for (size_t i = 0; i < mBatchSize; ++i) {
    mSessions.emplace_back(mModel.GetOperators());
  }
}

} // namespace colombia_supremo

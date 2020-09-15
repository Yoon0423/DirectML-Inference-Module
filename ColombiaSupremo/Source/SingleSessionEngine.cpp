// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "SingleSessionEngine.hpp"

namespace colombia_supremo {

SingleSessionEngine::SingleSessionEngine(const char *const modelFilePath)
    : InferenceEngine(modelFilePath, 1) {}

void SingleSessionEngine::Run(const std::vector<TensorRawData> &inputs,
                          std::vector<TensorRawData> &outputs) {
  mSessions[0].Run(inputs[0], outputs[0]);
}

} // namespace colombia_supremo

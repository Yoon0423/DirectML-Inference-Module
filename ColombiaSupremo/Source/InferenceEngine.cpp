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

void InferenceEngine::Run(const std::vector<TensorRawData> &inputs,
                          std::vector<TensorRawData> &outputs) {
  assert(inputs.size() == mBatchSize && outputs.size() == mBatchSize);

  const auto run = [](Session &session, const TensorRawData &input,
                      TensorRawData &output, std::promise<bool> &promise) {
    session.Run(input, output);
    promise.set_value(true);
  };

  std::vector<std::thread> threads;
  std::vector<std::future<bool>> futures;
  std::vector<std::promise<bool>> promises;
  for (size_t i = 0; i < mBatchSize; ++i) {
    promises.emplace_back(std::promise<bool>());
  }

  for (size_t i = 0; i < mBatchSize; ++i) {
    futures.emplace_back(move(promises[i].get_future()));

    threads.emplace_back(
        std::move(std::thread(run, std::ref(mSessions[i]), std::ref(inputs[i]),
                              std::ref(outputs[i]), ref(promises[i]))));
  }

  for (size_t i = 0; i < mBatchSize; ++i) {
    futures[i].wait();
  }

  for (size_t i = 0; i < mBatchSize; ++i) {
    threads[i].join();
  }
}

} // namespace colombia_supremo

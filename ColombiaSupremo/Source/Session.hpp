// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "Device.hpp"
#include "Operator.hpp"

namespace colombia_supremo {

class Session {
public:
  Session() = delete;
  Session(std::vector<std::shared_ptr<Operator>> operators);
  ~Session() = default;

  void Run(const TensorRawData &input, TensorRawData &output);

private:
  std::vector<std::shared_ptr<Operator>> mOperators;
  std::vector<std::shared_ptr<InOutTensor>> mIntermediateTensors;
  std::shared_ptr<UploadTensor> mUploadTensor;
  std::shared_ptr<ReadbackTensor> mReadbackTensor;

  std::shared_ptr<Device> mDevice;
};

} // namespace colombia_supremo

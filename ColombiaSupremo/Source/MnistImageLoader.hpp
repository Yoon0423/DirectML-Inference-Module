// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DataLoader.hpp"
#include "Tensor.hpp"

#include <vector>

namespace colombia_supremo {

class MnistImageLoader : public DataLoader<TensorRawData> {
public:
  MnistImageLoader() = delete;
  MnistImageLoader(const char *const filePath);
  ~MnistImageLoader() = default;

  TensorRawData LoadData(const size_t index) override;

private:
  std::vector<TensorRawData> mData;
  TensorShape mShape;
};

} // namespace colombia_supremo

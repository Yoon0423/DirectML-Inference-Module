// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DataLoader.hpp"
#include "Tensor.hpp"

#include <vector>

namespace colombia_supremo {

class MnistLabelLoader : public DataLoader<uint8_t> {
public:
  MnistLabelLoader() = delete;
  MnistLabelLoader(const char *const filePath);
  ~MnistLabelLoader() = default;

  uint8_t LoadData(const size_t index) override;

private:
  std::vector<uint8_t> mData;
};

} // namespace colombia_supremo

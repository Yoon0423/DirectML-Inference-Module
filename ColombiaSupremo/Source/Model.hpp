// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Operator.hpp"

#include <memory>
#include <string>
#include <vector>

namespace colombia_supremo {

class Model {
public:
  Model(const char *const filePath);

  std::vector<std::shared_ptr<Operator>> GetOperators();

private:
  TensorShape ReadTensorShapeFrom(std::fstream &fileStream);
  TensorRawData ReadTensorRawDataFrom(std::fstream &fileStream, const TensorShape shape);
  std::vector<uint32_t> ReadKernelShapeFrom(std::fstream &fileStream);
  std::vector<uint32_t> ReadStridesFrom(std::fstream &fileStream);
  std::vector<uint32_t> ReadPaddingsFrom(std::fstream &fileStream);

  std::string mFilePath;
  std::vector<std::shared_ptr<Operator>> mOperators;
};

} // namespace colombia_supremo

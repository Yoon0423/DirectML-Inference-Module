// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Operator.hpp"

namespace colombia_supremo {

class FullyConnectionOperator : public Operator {
public:
  FullyConnectionOperator() = delete;
  FullyConnectionOperator(const uint32_t inputLength,
                          const uint32_t outputLength,
                          const std::shared_ptr<WeightTensor> weightTensor);
  ~FullyConnectionOperator() = default;

  std::shared_ptr<WeightTensor> getWeightTensor();

  void InitBindingTables() override;

private:
  std::shared_ptr<WeightTensor> mWeightTensor;
};

} // namespace colombia_supremo
// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Operator.hpp"

namespace colombia_supremo {

class MaxPoolOperator : public Operator {
public:
  MaxPoolOperator() = delete;
  MaxPoolOperator(TensorShape inputShape, TensorShape outputShape,
                  const uint32_t stride, const uint32_t padding, const uint32_t kernelSize);
  ~MaxPoolOperator() = default;

  void InitBindingTables() override;

private:
  uint32_t mStride;
  uint32_t mPadding;
};

} // namespace colombia_supremo

// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Operator.hpp"

namespace colombia_supremo {

class ConvolutionOperator : public Operator {
public:
  ConvolutionOperator() = delete;
  ConvolutionOperator(TensorShape inputShape, TensorShape outputShape,
                      const std::shared_ptr<WeightTensor> weightTensor,
                      const std::shared_ptr<WeightTensor> biasTensor,
                      const uint32_t stride, const uint32_t padding,
                      const bool isReluActivated);
  ~ConvolutionOperator() = default;

  std::shared_ptr<WeightTensor> getWeightTensor();

  void InitBindingTables() override;

private:
  std::shared_ptr<WeightTensor> mWeightTensor;
  std::shared_ptr<WeightTensor> mBiasTensor;

  uint32_t mStride;
  uint32_t mPadding;
};

} // namespace colombia_supremo

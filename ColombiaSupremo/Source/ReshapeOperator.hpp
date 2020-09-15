// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Operator.hpp"

namespace colombia_supremo {

class ReshapeOperator : public Operator {
public:
  ReshapeOperator() = delete;
  ReshapeOperator(TensorShape inputShape, TensorShape outputShape);
  ~ReshapeOperator() = default;

  void Run(std::shared_ptr<Device> device, std::shared_ptr<Tensor> inputTensor,
           std::shared_ptr<Tensor> outputTensor) override;
  void InitBindingTables() override;
};

} // namespace colombia_supremo

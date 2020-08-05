// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <DirectML.h>
#include <winrt/base.h>

#include <memory>

#include "Tensor.hpp"

namespace colombia_supremo {

class Operator {
public:
  Operator() = delete;
  Operator(TensorShape inputShape, TensorShape outputShape);
  ~Operator() = default;

  std::shared_ptr<InOutTensor> getInputTensor();
  std::shared_ptr<InOutTensor> getOutputTensor();
  winrt::com_ptr<IDMLCompiledOperator> getCompiledOperator(); // remove?
  winrt::com_ptr<IDMLBindingTable> getExecBindingTable();     // remove?
  void Run(std::shared_ptr<UploadTensor> uploadTensor,
           std::shared_ptr<ReadbackTensor> readbackTensor);

  virtual void InitBindingTables() = 0;

  std::shared_ptr<UploadTensor> CreateNewUploadTensor();
  std::shared_ptr<ReadbackTensor> CreateNewReadbackTensor();

protected:
  std::shared_ptr<InOutTensor> mInputTensor;
  std::shared_ptr<InOutTensor> mOutputTensor;
  winrt::com_ptr<IDMLCompiledOperator> mCompiledOperator;

  winrt::com_ptr<ID3D12DescriptorHeap> mInitDescriptorHeap;
  winrt::com_ptr<ID3D12DescriptorHeap> mExecDescriptorHeap;
  winrt::com_ptr<IDMLBindingTable> mInitBindingTable;
  winrt::com_ptr<IDMLBindingTable> mExecBindingTable;
  winrt::com_ptr<ID3D12Resource> mPersistentResource;
};

} // namespace colombia_supremo

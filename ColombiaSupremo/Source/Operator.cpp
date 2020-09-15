// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <cstring>

#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"
#include "Operator.hpp"

namespace colombia_supremo {

Operator::Operator(TensorShape inputShape, TensorShape outputShape)
    : mInputTensor(std::make_shared<InOutTensor>(std::move(inputShape))),
      mOutputTensor(std::make_shared<InOutTensor>(std::move(outputShape))) {}

std::shared_ptr<InOutTensor> Operator::getInputTensor() { return mInputTensor; }

std::shared_ptr<InOutTensor> Operator::getOutputTensor() {
  return mOutputTensor;
}

winrt::com_ptr<IDMLCompiledOperator> Operator::getCompiledOperator() {
  return mCompiledOperator;
}

winrt::com_ptr<IDMLBindingTable> Operator::getExecBindingTable() {
  return mExecBindingTable;
}

void Operator::Run(std::shared_ptr<Device> device,
                   std::shared_ptr<Tensor> inputTensor,
                   std::shared_ptr<Tensor> outputTensor) {
  // TODO: is it going to be a huge overhead?
  ID3D12DescriptorHeap *descriptorHeaps[] = {mExecDescriptorHeap.get()};
  device->mCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                                  descriptorHeaps);

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             inputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_SOURCE));

  device->mCommandList->ResourceBarrier(
      1,
      &CD3DX12_RESOURCE_BARRIER::Transition(
          mInputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_COPY_DEST));

  device->mCommandList->CopyResource(mInputTensor->getBufferPtr(),
                                           inputTensor->getBufferPtr());

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             mInputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  device->mCommandRecorder->RecordDispatch(device->mCommandList.get(),
                                           mCompiledOperator.get(),
      mExecBindingTable.get());

  device->mCommandList->ResourceBarrier(
      1,
      &CD3DX12_RESOURCE_BARRIER::Transition(
          outputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_COPY_DEST));

  device->mCommandList->ResourceBarrier(
      1,
      &CD3DX12_RESOURCE_BARRIER::Transition(
          mOutputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_COPY_SOURCE));

  device->mCommandList->CopyResource(outputTensor->getBufferPtr(),
                                           mOutputTensor->getBufferPtr());

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             outputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             mOutputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_COPY_SOURCE,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  //device->CloseExecuteResetWait();
}

std::shared_ptr<UploadTensor> Operator::CreateNewUploadTensor() {
  return std::make_shared<UploadTensor>(mInputTensor->GetShape());
}

std::shared_ptr<ReadbackTensor> Operator::CreateNewReadbackTensor() {
  return std::make_shared<ReadbackTensor>(mOutputTensor->GetShape());
}

} // namespace colombia_supremo

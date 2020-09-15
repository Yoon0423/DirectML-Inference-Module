// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "ReshapeOperator.hpp"
#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"
#include "DirectMLHelperFunction.hpp"

namespace colombia_supremo {

ReshapeOperator::ReshapeOperator(TensorShape inputShape,
                                 TensorShape outputShape)
    : Operator(std::move(inputShape), std::move(outputShape)) {
  auto defaultDevice = DeviceManager::getInstance().GetDefault();

  defaultDevice->mCommandList->ResourceBarrier(
      1,
      &CD3DX12_RESOURCE_BARRIER::Transition(
          mInputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_COPY_SOURCE));

  defaultDevice->mCommandList->ResourceBarrier(
      1,
      &CD3DX12_RESOURCE_BARRIER::Transition(
          mOutputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_COPY_DEST));

  defaultDevice->CloseExecuteResetWait();
}

void ReshapeOperator::Run(std::shared_ptr<Device> device,
                          std::shared_ptr<Tensor> inputTensor,
                          std::shared_ptr<Tensor> outputTensor) {
  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             inputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_SOURCE));

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(mInputTensor->getBufferPtr(),
                                               D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               D3D12_RESOURCE_STATE_COPY_DEST));

  device->mCommandList->CopyResource(mInputTensor->getBufferPtr(),
                                     inputTensor->getBufferPtr());

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             mInputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_COPY_SOURCE));

  device->mCommandList->CopyResource(mOutputTensor->getBufferPtr(),
                                     mInputTensor->getBufferPtr());

  device->mCommandList->ResourceBarrier(
      1,
      &CD3DX12_RESOURCE_BARRIER::Transition(
          outputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_COPY_DEST));

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             mOutputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_COPY_SOURCE));

  device->mCommandList->CopyResource(outputTensor->getBufferPtr(),
                                     mOutputTensor->getBufferPtr());

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             outputTensor->getBufferPtr(), D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  device->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(mOutputTensor->getBufferPtr(),
                                               D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               D3D12_RESOURCE_STATE_COPY_DEST));

  device->CloseExecuteResetWait();
}

void ReshapeOperator::InitBindingTables() {
  // don't need to be implemented
}

} // namespace colombia_supremo

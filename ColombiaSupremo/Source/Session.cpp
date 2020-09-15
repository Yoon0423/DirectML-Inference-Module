// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <cassert>

#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"
#include "Session.hpp"

namespace colombia_supremo {

Session::Session(std::vector<std::shared_ptr<Operator>> operators)
    : mOperators(std::move(operators)) {
  assert(operators.size() == 0, "Session must have at least one operator");

  mIntermediateTensors.reserve(mOperators.size() + 1);

  mUploadTensor = mOperators[0]->CreateNewUploadTensor();

  mReadbackTensor =
      mOperators[mOperators.size() - 1]->CreateNewReadbackTensor();

  mIntermediateTensors.emplace_back(std::make_shared<InOutTensor>(
      mOperators[0]->getInputTensor()->GetShape()));

  for (size_t i = 0; i < mOperators.size(); ++i) {
    mIntermediateTensors.emplace_back(std::make_shared<InOutTensor>(
        mOperators[i]->getOutputTensor()->GetShape()));
  }

  // get device
  mDevice = DeviceManager::getInstance().CreateNewDevice();
}

void Session::Run(const TensorRawData &input, TensorRawData &output) {
  const size_t operatorCount = mOperators.size();

  mUploadTensor->ReadFromData(input);

  mDevice->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             mIntermediateTensors[0]->getBufferPtr(),
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_DEST));

  mDevice->mCommandList->CopyResource(
      mIntermediateTensors[0]->getBufferPtr(), mUploadTensor->getBufferPtr());

  mDevice->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             mIntermediateTensors[0]->getBufferPtr(),
             D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  //mDevice->CloseExecuteResetWait();

  for (size_t i = 0; i < operatorCount; ++i) {
    mOperators[i]->Run(mDevice, mIntermediateTensors[i], mIntermediateTensors[i + 1]);
  }

  mDevice->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             mIntermediateTensors[operatorCount]->getBufferPtr(),
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
             D3D12_RESOURCE_STATE_COPY_SOURCE));

  mDevice->mCommandList->CopyResource(
      mReadbackTensor->getBufferPtr(),
      mIntermediateTensors[operatorCount]->getBufferPtr());

  mDevice->mCommandList->ResourceBarrier(
      1, &CD3DX12_RESOURCE_BARRIER::Transition(
             mIntermediateTensors[operatorCount]->getBufferPtr(),
             D3D12_RESOURCE_STATE_COPY_SOURCE,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

  mDevice->CloseExecuteResetWait();

  mReadbackTensor->WriteToData(output);
}

} // namespace colombia_supremo

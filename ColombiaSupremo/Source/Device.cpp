// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"

namespace colombia_supremo {

Device::Device(const uint32_t id, winrt::com_ptr<ID3D12Device> d3D12Device,
               winrt::com_ptr<ID3D12CommandQueue> commandQueue,
               winrt::com_ptr<ID3D12CommandAllocator> commandAllocator,
               winrt::com_ptr<ID3D12GraphicsCommandList> commandList,
               winrt::com_ptr<IDMLDevice> dmlDevice,
               winrt::com_ptr<IDMLCommandRecorder> commandRecorder)
    : mId(id), mD3D12Device(d3D12Device), mCommandQueue(commandQueue),
      mCommandAllocator(commandAllocator), mCommandList(commandList),
      mDMLDevice(dmlDevice), mCommandRecorder(commandRecorder) {}

void Device::CloseExecuteResetWait() {
  d3d12_helper::CloseExecuteResetWait(mD3D12Device, mCommandQueue,
                                      mCommandAllocator, mCommandList);
}

} // namespace colombia_supremo

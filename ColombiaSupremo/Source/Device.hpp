// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <DirectML.h>
#include <stdint.h>
#include <winrt/base.h>

namespace colombia_supremo {

class Device {
public:
  Device() = delete;
  Device(const uint32_t id, winrt::com_ptr<ID3D12Device> d3D12Device,
         winrt::com_ptr<ID3D12CommandQueue> commandQueue,
         winrt::com_ptr<ID3D12CommandAllocator> commandAllocator,
         winrt::com_ptr<ID3D12GraphicsCommandList> commandList,
         winrt::com_ptr<IDMLDevice> dmlDevice,
         winrt::com_ptr<IDMLCommandRecorder> commandRecorder);
  ~Device() = default;

  void CloseExecuteResetWait();

public: // TODO private
  const uint32_t mId;

  winrt::com_ptr<ID3D12Device> mD3D12Device;
  winrt::com_ptr<ID3D12CommandQueue> mCommandQueue;
  winrt::com_ptr<ID3D12CommandAllocator> mCommandAllocator;
  winrt::com_ptr<ID3D12GraphicsCommandList> mCommandList;

  winrt::com_ptr<IDMLDevice> mDMLDevice;
  winrt::com_ptr<IDMLCommandRecorder> mCommandRecorder;
};

} // namespace colombia_supremo

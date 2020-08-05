// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <DirectML.h>
#include <winrt/base.h>

namespace colombia_supremo {

class DeviceManager {
public:
  static DeviceManager &getInstance();

  // TODO: private ?
  winrt::com_ptr<ID3D12Device> mD3D12Device;                // be a single
  winrt::com_ptr<ID3D12CommandQueue> mCommandQueue;         // be a single(?)
  winrt::com_ptr<ID3D12CommandAllocator> mCommandAllocator; // per thread
  winrt::com_ptr<ID3D12GraphicsCommandList> mCommandList;   // per thread

  winrt::com_ptr<IDMLDevice> mDMLDevice;                // be a single
  winrt::com_ptr<IDMLCommandRecorder> mCommandRecorder; // per thread

private:
  DeviceManager();

public:
  // build for C++11~
  DeviceManager(DeviceManager const &obj) = delete;
  void operator=(DeviceManager const &obj) = delete;
};

} // namespace colombia_supremo

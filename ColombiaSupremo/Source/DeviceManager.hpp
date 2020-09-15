// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Device.hpp"

#include <DirectML.h>
#include <memory>
#include <winrt/base.h>

namespace colombia_supremo {

class DeviceManager {
public:
  static DeviceManager &getInstance();
  
  uint32_t mDeviceCount; // TODO: private ?

  winrt::com_ptr<ID3D12Device> mD3D12Device;
  winrt::com_ptr<ID3D12CommandQueue> mCommandQueue;

  winrt::com_ptr<IDMLDevice> mDMLDevice;

  std::shared_ptr<Device> CreateNewDevice();
  std::shared_ptr<Device> GetDefault();

private:
  DeviceManager();

public:
  // build for C++11~
  DeviceManager(DeviceManager const &obj) = delete;
  void operator=(DeviceManager const &obj) = delete;
};

} // namespace colombia_supremo

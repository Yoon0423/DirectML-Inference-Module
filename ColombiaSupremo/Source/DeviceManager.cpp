// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"
#include "DirectMLHelperFunction.hpp"

namespace colombia_supremo {

DeviceManager &colombia_supremo::DeviceManager::getInstance() {
  static DeviceManager instance;

  return instance;
}

std::shared_ptr<Device> DeviceManager::CreateNewDevice() {
  auto commandAllocator = d3d12_helper::CreateCommandAllocator(mD3D12Device);
  auto commandList =
      d3d12_helper::CreateCommandList(mD3D12Device, commandAllocator);
  auto commandRecorder = dml_helper::CreateCommandRecorder(mDMLDevice);

  auto device = std::make_shared<Device>(
      mDeviceCount++, mD3D12Device, mCommandQueue, commandAllocator,
      commandList, mDMLDevice, commandRecorder);

  return device;
}

std::shared_ptr<Device> DeviceManager::GetDefault() {
  static std::shared_ptr<Device> device = CreateNewDevice();

  return device;
}

DeviceManager::DeviceManager() : mDeviceCount(0) {
  mD3D12Device = d3d12_helper::CreateDevice();
  mCommandQueue = d3d12_helper::CreateCommandQueue(mD3D12Device);
  mDMLDevice = dml_helper::CreateDevice(mD3D12Device);
}

} // namespace colombia_supremo

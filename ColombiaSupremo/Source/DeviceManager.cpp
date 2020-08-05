// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"

namespace colombia_supremo {

DeviceManager &colombia_supremo::DeviceManager::getInstance() {
  static DeviceManager instance;

  return instance;
}

DeviceManager::DeviceManager() {
  // functionalize to InitializeD3D12Resources() ?
  d3d12_helper::InitializeDirect3D12(mD3D12Device, mCommandQueue,
                                     mCommandAllocator, mCommandList);

  // functionalize to InitializeDMLResources() ?
  auto dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;

#if defined(_DEBUG)
  dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

  winrt::check_hresult(DMLCreateDevice(mD3D12Device.get(), dmlCreateDeviceFlags,
                                       __uuidof(mDMLDevice),
                                       mDMLDevice.put_void()));

  winrt::check_hresult(mDMLDevice->CreateCommandRecorder(
      __uuidof(mCommandRecorder), mCommandRecorder.put_void()));
}

} // namespace colombia_supremo

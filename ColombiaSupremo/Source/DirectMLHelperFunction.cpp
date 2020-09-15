// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "DirectMLHelperFunction.hpp"
#include <cassert>

namespace colombia_supremo::dml_helper {

winrt::com_ptr<IDMLDevice>
CreateDevice(winrt::com_ptr<ID3D12Device> d3D12Device) {
  winrt::com_ptr<IDMLDevice> device;

  auto dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;

#if defined(_DEBUG)
  dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

  winrt::check_hresult(DMLCreateDevice(d3D12Device.get(), dmlCreateDeviceFlags,
                                       __uuidof(device), device.put_void()));

  return device;
}

winrt::com_ptr<IDMLCommandRecorder>
CreateCommandRecorder(winrt::com_ptr<IDMLDevice> device) {
  winrt::com_ptr<IDMLCommandRecorder> commandRecorder;

  winrt::check_hresult(device->CreateCommandRecorder(
      __uuidof(commandRecorder), commandRecorder.put_void()));

  return commandRecorder;
}

uint64_t CalculateBufferTensorSize(const TensorShape &tensorShape) {
  uint64_t tensorElementCount = 1;

  for (const auto &element : tensorShape) {
    tensorElementCount *= static_cast<uint64_t>(element);
  }

  const uint64_t elementSizeInBytes = 4; // we support only 32-bit float

  return tensorElementCount * elementSizeInBytes;
}

TensorShape CalculateStrides(const TensorShape &tensorShape) {
  TensorShape stride;
  stride.reserve(tensorShape.size());

  // only NCHW allowed
  stride.emplace_back(tensorShape[1] * tensorShape[2] * tensorShape[3]);
  stride.emplace_back(tensorShape[2] * tensorShape[3]);
  stride.emplace_back(tensorShape[3]);
  stride.emplace_back(1);

  return stride;
}

} // namespace colombia_supremo::dml_helper

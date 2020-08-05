// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "Tensor.hpp"
#include "DeviceManager.hpp"
#include "Direct3D12HelperFunction.hpp"
#include "DirectMLHelperFunction.hpp"
#include "d3dx12.h"

namespace colombia_supremo {

Tensor::Tensor(TensorShape shape) : mShape(std::move(shape)) {
  mTensorBufferSize = dml_helper::CalculateBufferTensorSize(mShape);
}

ID3D12Resource *Tensor::getBufferPtr() { return mBuffer.get(); }

uint64_t Tensor::getTensorBufferSize() { return mTensorBufferSize; }

TensorShape &Tensor::getShapeRef() { return mShape; }

TensorShape Tensor::GetShape() { return mShape; }

InOutTensor::InOutTensor(TensorShape shape) : Tensor(shape) {
  auto &deviceManager = DeviceManager::getInstance();

  // create buffer
  winrt::check_hresult(deviceManager.mD3D12Device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(
          getTensorBufferSize(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(mBuffer),
      mBuffer.put_void()));
}

WeightTensor::WeightTensor(const TensorRawData &weights, TensorShape shape)
    : Tensor(shape) {
  auto &deviceManager = DeviceManager::getInstance();

  const auto tensorResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
      getTensorBufferSize(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

  winrt::check_hresult(deviceManager.mD3D12Device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
      &tensorResourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr,
      __uuidof(mBuffer), mBuffer.put_void()));

  winrt::com_ptr<ID3D12Resource> uploadBuffer;
  winrt::check_hresult(deviceManager.mD3D12Device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(getTensorBufferSize()),
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, __uuidof(uploadBuffer),
      uploadBuffer.put_void()));

  uint8_t *ptr;
  CD3DX12_RANGE range(0, 0);
  uploadBuffer->Map(0, &range, reinterpret_cast<void **>(&ptr));
  memcpy(ptr, reinterpret_cast<const uint8_t *>(weights.data()),
         weights.size() * 4);
  uploadBuffer->Unmap(0, nullptr);

  deviceManager.mCommandList->CopyResource(getBufferPtr(), uploadBuffer.get());

  d3d12_helper::CloseExecuteResetWait(
      deviceManager.mD3D12Device, deviceManager.mCommandQueue,
      deviceManager.mCommandAllocator, deviceManager.mCommandList);
}

UploadTensor::UploadTensor(TensorShape shape) : Tensor(shape) {
  auto &deviceManager = DeviceManager::getInstance();

  winrt::check_hresult(deviceManager.mD3D12Device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(getTensorBufferSize()),
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, __uuidof(mBuffer),
      mBuffer.put_void()));
}

void UploadTensor::ReadFromData(const TensorRawData &data) {
  uint8_t *bytePtr;
  const static CD3DX12_RANGE range(0, 0);
  getBufferPtr()->Map(0, &range, reinterpret_cast<void **>(&bytePtr));
  memcpy(bytePtr, reinterpret_cast<const uint8_t *>(data.data()),
         data.size() * sizeof(data[0]));
  getBufferPtr()->Unmap(0, nullptr);
}

ReadbackTensor::ReadbackTensor(TensorShape shape) : Tensor(shape) {
  auto &deviceManager = DeviceManager::getInstance();

  winrt::check_hresult(deviceManager.mD3D12Device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
      &CD3DX12_RESOURCE_DESC::Buffer(getTensorBufferSize()),
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(mBuffer),
      mBuffer.put_void()));
}

void ReadbackTensor::WriteToData(TensorRawData &data) {
  uint8_t *bytePtr;
  const size_t byteSize = data.size() * sizeof(data[0]);
  const static CD3DX12_RANGE range(0, byteSize);
  getBufferPtr()->Map(0, &range, reinterpret_cast<void **>(&bytePtr));
  memcpy(reinterpret_cast<uint8_t *>(data.data()), bytePtr, byteSize);
  getBufferPtr()->Unmap(0, nullptr);
}

} // namespace colombia_supremo

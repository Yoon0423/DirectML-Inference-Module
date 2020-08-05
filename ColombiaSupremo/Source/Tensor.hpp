// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <DirectML.h>
#include <cstdint>
#include <vector>
#include <winrt/base.h>

namespace colombia_supremo {

using TensorSizes = std::vector<uint32_t>; // remove
using TensorShape = std::vector<uint32_t>;
using TensorRawData = std::vector<float>;
using TensorData = std::vector<float>; // remove

// remove
struct TensorDescriptor {
  TensorSizes tensorSizes;
  TensorData tensorData; // TODO: only 32-bit float?
};

class Tensor {
public:
  ID3D12Resource *getBufferPtr();
  uint64_t getTensorBufferSize();
  TensorShape &getShapeRef();
  TensorShape GetShape();

protected:
  Tensor() = delete;
  Tensor(TensorShape shape);
  ~Tensor() = default;

  winrt::com_ptr<ID3D12Resource> mBuffer;
  uint64_t mTensorBufferSize;
  TensorShape mShape;
};

class InOutTensor : public Tensor {
public:
  InOutTensor() = delete;
  InOutTensor(TensorShape shape);
  ~InOutTensor() = default;
};

class WeightTensor : public Tensor {
public:
  WeightTensor() = delete;
  WeightTensor(const TensorRawData &weights, TensorShape shape);
  ~WeightTensor() = default;
};

class UploadTensor : public Tensor {
public:
  UploadTensor() = delete;
  UploadTensor(TensorShape shape);
  ~UploadTensor() = default;

  void ReadFromData(const TensorRawData &data);
};

class ReadbackTensor : public Tensor {
public:
  ReadbackTensor() = delete;
  ReadbackTensor(TensorShape shape);
  ~ReadbackTensor() = default;

  void WriteToData(TensorRawData &data);
};

} // namespace colombia_supremo

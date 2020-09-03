// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "Model.hpp"
#include "ConvolutionOperator.hpp"
#include "FullyConnectionOperator.hpp"
#include "MaxPoolOperator.hpp"
#include "ReshapeOperator.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace colombia_supremo {

Model::Model(const char *const filePath) : mFilePath(std::string(filePath)) {
  std::fstream file(mFilePath.c_str());

  if (file.is_open() == false) {
    std::runtime_error("cannot open file: " + mFilePath);
    abort();
  }

  std::string str;
  while (std::getline(file, str)) {
    if (str == "Conv") {
      const TensorShape inputShape = ReadTensorShapeFrom(file);
      const TensorShape outputShape = ReadTensorShapeFrom(file);

      const TensorShape weightShape = ReadTensorShapeFrom(file);
      TensorRawData weightData = ReadTensorRawDataFrom(file, weightShape);

      const TensorShape biasShape = ReadTensorShapeFrom(file);
      TensorRawData biasData = ReadTensorRawDataFrom(file, biasShape);

      std::string activation;
      std::getline(file, activation);
      bool isReluActivation = false;
      if (activation == "Relu") {
        isReluActivation = true;
      }

      const auto kernelShape = ReadKernelShapeFrom(file);
      const auto strides = ReadStridesFrom(file);
      const auto paddings = ReadPaddingsFrom(file);

      mOperators.emplace_back(std::make_shared<ConvolutionOperator>(
          inputShape, outputShape,
          std::make_shared<WeightTensor>(weightData, weightShape),
          std::make_shared<WeightTensor>(biasData, biasShape), strides[0],
          paddings[0], isReluActivation));
    } else if (str == "MaxPool") {
      const TensorShape inputShape = ReadTensorShapeFrom(file);
      const TensorShape outputShape = ReadTensorShapeFrom(file);

      const auto kernelShape = ReadKernelShapeFrom(file);
      const auto strides = ReadStridesFrom(file);
      const auto paddings = ReadPaddingsFrom(file);

      mOperators.emplace_back(std::make_shared<MaxPoolOperator>(
          inputShape, outputShape, strides[0], paddings[0], kernelShape[0]));
    } else if (str == "FullyConnection") {
      const TensorShape inputShape = ReadTensorShapeFrom(file);
      const TensorShape outputShape = ReadTensorShapeFrom(file);

      const TensorShape weightShape = ReadTensorShapeFrom(file);
      TensorRawData weightData = ReadTensorRawDataFrom(file, weightShape);

      const TensorShape biasShape = ReadTensorShapeFrom(file);
      TensorRawData biasData = ReadTensorRawDataFrom(file, biasShape);

      mOperators.emplace_back(std::make_shared<FullyConnectionOperator>(
          inputShape[3], outputShape[3],
          std::make_shared<WeightTensor>(weightData, weightShape),
          std::make_shared<TensorRawData>(biasData)));
    } else if (str == "Reshape") {
      const TensorShape inputShape = ReadTensorShapeFrom(file);
      const TensorShape outputShape = ReadTensorShapeFrom(file);

      mOperators.emplace_back(
          std::make_shared<ReshapeOperator>(inputShape, outputShape));
    } else {
      std::runtime_error("unsupported operator or invalid string: " + str);
      abort();
    }
  }
  file.close();
}

std::vector<std::shared_ptr<Operator>> Model::GetOperators() {
  return mOperators;
}

TensorShape Model::ReadTensorShapeFrom(std::fstream &fileStream) {
  TensorShape shape({0, 0, 0, 0});

  for (size_t i = 0; i < shape.size(); ++i) {
    fileStream >> shape[i];
  }
  fileStream.ignore();

  return shape;
}

TensorRawData Model::ReadTensorRawDataFrom(std::fstream &fileStream,
                                           const TensorShape shape) {
  const size_t size =
      static_cast<size_t>(shape[0] * shape[1] * shape[2] * shape[3]);

  TensorRawData data;
  data.reserve(size);

  for (size_t i = 0; i < size; ++i) {
    static float value;
    fileStream >> value;
    data.emplace_back(value);
  }

  fileStream.ignore();

  return data;
}

std::vector<uint32_t> Model::ReadKernelShapeFrom(std::fstream &fileStream) {
  std::vector<uint32_t> result;
  result.reserve(2);

  uint32_t value;

  fileStream >> value;
  result.emplace_back(value);

  fileStream >> value;
  result.emplace_back(value);

  fileStream.ignore();

  return result;
}

std::vector<uint32_t> Model::ReadStridesFrom(std::fstream &fileStream) {
  std::vector<uint32_t> result;
  result.reserve(2);

  uint32_t value;

  fileStream >> value;
  result.emplace_back(value);

  fileStream >> value;
  result.emplace_back(value);

  fileStream.ignore();

  return result;
}

std::vector<uint32_t> Model::ReadPaddingsFrom(std::fstream &fileStream) {
  std::vector<uint32_t> result;
  result.reserve(4);

  uint32_t value;

  fileStream >> value;
  result.emplace_back(value);

  fileStream >> value;
  result.emplace_back(value);

  fileStream >> value;
  result.emplace_back(value);

  fileStream >> value;
  result.emplace_back(value);

  fileStream.ignore();

  return result;
}

} // namespace colombia_supremo

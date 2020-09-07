// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <InferenceEngine.hpp>
#include <MnistImageLoader.hpp>
#include <MnistLabelLoader.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace colombia_supremo;

void setZeroTensorData(vector<TensorRawData> &tensors);

int main(int argc, char **argv) {
  cout << "TestInferenceEngine" << endl;

  const size_t batchSize = 3;

  InferenceEngine engine("../Model/converted_mnist.txt", batchSize);

  const size_t width = 28;
  const size_t labelsCount = 10;
  vector<TensorRawData> inputs;
  vector<TensorRawData> outputs;
  vector<uint8_t> answers;

  for (size_t i = 0; i < batchSize; ++i) {
    TensorRawData input;
    input.reserve(width * width);
    for (size_t j = 0; j < width * width; ++j) {
      input.emplace_back(0.f);
    }
    inputs.emplace_back(input);

    TensorRawData output;
    for (size_t j = 0; j < labelsCount; ++j) {
      output.emplace_back(0.f);
    }
    outputs.emplace_back(output);

    answers.emplace_back(0);
  }

  MnistImageLoader imageLoader("../Model/t10k-images.idx3-ubyte");
  MnistLabelLoader labelLoader("../Model/t10k-labels.idx1-ubyte");

  for (size_t i = 0; i < 3; ++i) {
    inputs[i] = imageLoader.LoadData(i);
    answers[i] = labelLoader.LoadData(i);
  }

  engine.Run(inputs, outputs);

  for (size_t i = 0; i < batchSize; ++i) {
    const auto& output = outputs[i];

    size_t result = 0;
    float value = output[0];
    for (size_t j = 0; j < 10; ++j) {
      if (output[j] > value) {
        result = j;
        value = output[j];
      }
    }

    cout << "inferenced output: " << result << " | answer: " << static_cast<int>(answers[i]) << endl;
  }

  return 0;
}

void setZeroTensorData(vector<TensorRawData> &tensors) {}

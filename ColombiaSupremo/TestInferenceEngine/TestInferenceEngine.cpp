// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <InferenceEngine.hpp>
#include <MnistImageLoader.hpp>
#include <MnistLabelLoader.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace chrono;
using namespace colombia_supremo;

void setZeroTensorData(vector<TensorRawData> &tensors);

int main(int argc, char **argv) {
  cout << "TestOnnxruntime" << endl;

  const size_t batchSize = 5;

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

  double elapsedNanoseconds = 0.;

  for (size_t i = 0; i < 100; i += batchSize) {
    for (size_t j = 0; j < batchSize; ++j) {
      inputs[j] = imageLoader.LoadData(i + j);
      answers[j] = labelLoader.LoadData(i + j);
    }

    const auto t_start = high_resolution_clock::now();
    engine.Run(inputs, outputs);
    const auto t_end = high_resolution_clock::now();

    elapsedNanoseconds += static_cast<double>(
        duration_cast<nanoseconds>(t_end - t_start).count());

    for (size_t j = 0; j < batchSize; ++j) {
      const auto &output = outputs[j];

      size_t result = 0;
      float value = output[0];
      for (size_t k = 0; k < 10; ++k) {
        if (output[k] > value) {
          result = k;
          value = output[k];
        }
      }

      //cout << "inferenced output: " << result
      //     << " | answer: " << static_cast<int>(answers[j]) << endl;
    }
  }

  cout << "elapsed time: "
       << static_cast<long long>(elapsedNanoseconds / 1000000.0) << endl;

  return 0;
}

void setZeroTensorData(vector<TensorRawData> &tensors) {}

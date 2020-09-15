// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <MnistImageLoader.hpp>
#include <MnistLabelLoader.hpp>
#include <Model.hpp>
#include <Session.hpp>
#include <iostream>

using namespace std;
using namespace colombia_supremo;

int main(int argc, char **argv) {
  cout << "TestModel" << endl;

  cout << "load mnist test images and labels..." << endl;
  MnistImageLoader imageLoader("../Model/t10k-images.idx3-ubyte");
  MnistLabelLoader labelLoader("../Model/t10k-labels.idx1-ubyte");

  Model model("../Model/converted_mnist.txt");
  Session session(model.GetOperators());

  const size_t outputSize = 10;
  TensorRawData output;
  output.reserve(outputSize);
  for (size_t i = 0; i < outputSize; ++i) {
    output.emplace_back(0.f);
  }

  const size_t iters = 10;
  for (size_t i = 0; i < iters; ++i) {
    const auto input = imageLoader.LoadData(i);
    const auto label = labelLoader.LoadData(i);

    session.Run(input, output);

    size_t answer = 0;
    float value = output[0];
    for (size_t j = 0; j < 10; ++j) {
      if (output[j] > value) {
        answer = j;
        value = output[j];
      }
    }

    for (auto element : output) {
      cout << element << " ";
    }
    cout << endl;

    cout << "answer: " << answer << " | " << static_cast<int>(label) << endl;

    for (size_t i = 0; i < outputSize; ++i) {
      output[i] = 0.f;
    }
  }

  return 0;
}

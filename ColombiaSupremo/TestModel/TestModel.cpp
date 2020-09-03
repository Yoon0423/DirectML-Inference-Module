// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <Model.hpp>
#include <Session.hpp>
#include <iostream>

using namespace std;
using namespace colombia_supremo;

int main(int argc, char **argv) {

  Model model("converted_mnist.txt");
  Session session(model.GetOperators());

  const size_t width = 28;
  TensorRawData input;
  input.reserve(width * width);
  for (size_t i = 0; i < width * width; ++i) {
    input.emplace_back(0.f);
  }

  const size_t outputSize = 10;
  TensorRawData output;
  output.reserve(outputSize);
  for (size_t i = 0; i < outputSize; ++i) {
    output.emplace_back(0.f);
  }

  session.Run(input, output);

  for (const auto element : output) {
    cout << element << " ";
  }
  cout << endl;

  return 0;
}

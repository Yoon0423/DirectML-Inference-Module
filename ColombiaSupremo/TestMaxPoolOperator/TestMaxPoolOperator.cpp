// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <MaxPoolOperator.hpp>
#include <Session.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace colombia_supremo;

int main(int argc, char **argv) {
  cout << "TestMaxPoolOperator" << endl;
  const TensorShape inputShape({1, 2, 4, 4});
  const TensorShape outputShape({1, 2, 2, 2});
  const uint32_t stride = 2;
  const uint32_t padding = 0;
  const uint32_t kernelSize = 2;

  vector<shared_ptr<Operator>> operators;
  operators.emplace_back(make_shared<MaxPoolOperator>(
      inputShape, outputShape, stride, padding, kernelSize));

  Session session(operators);

  TensorRawData input;
  input.reserve(2 * 4 * 4);
  {
    float v = 1.f;
    for (size_t c = 0; c < 2; ++c) {
      for (size_t h = 0; h < 4; ++h) {
        for (size_t w = 0; w < 4; ++w) {
          input.emplace_back(v++);
        }
      }
    }
  }

  TensorRawData output;
  output.reserve(2 * 2 * 2);
  for (size_t i = 0; i < 8; ++i) {
    output.emplace_back(0.f);
  }

  session.Run(input, output);

  for (size_t c = 0; c < 2; ++c) {
    cout << "channel " << c << endl;
    const size_t HW = 4;
    for (size_t h = 0; h < 2; ++h) {
      const size_t W = 2;
      for (size_t w = 0; w < 2; ++w) {
        cout << output[HW * c + W * h + w] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;

  return 0;
}

// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <FullyConnectionOperator.hpp>
#include <ReshapeOperator.hpp>
#include <Session.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace colombia_supremo;

int main(int argc, char **argv) {
  cout << "TestReshapeOperator" << endl;

  const TensorShape shape_0({1, 1, 2, 2});
  const TensorShape shape_1({1, 1, 1, 4});
  const uint32_t size_1 = 4;
  const uint32_t size_2 = 3;
  TensorRawData weightData;
  TensorShape weightShape({1, 1, size_1, size_2});
  {
    for (uint32_t r = 0; r < size_1; ++r) {
      float v = 1.f;
      for (uint32_t c = 0; c < size_2; ++c) {
        weightData.emplace_back(v++);
      }
    }
  }

  vector<shared_ptr<Operator>> operators;
  operators.emplace_back(make_shared<ReshapeOperator>(shape_0, shape_1));
  operators.emplace_back(make_shared<FullyConnectionOperator>(
      size_1, size_2, make_shared<WeightTensor>(weightData, weightShape),
      nullptr));

  Session session(operators);

  TensorRawData input({1.f, 2.f, 3.f, 4.f});
  TensorRawData output({0.f, 0.f, 0.f});

  session.Run(input, output);

  for (const auto &element : output) {
    cout << element << " ";
  }
  cout << endl;
  return 0;
}

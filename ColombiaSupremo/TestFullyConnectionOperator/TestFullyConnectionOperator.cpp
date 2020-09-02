#include <FullyConnectionOperator.hpp>
#include <Session.hpp>
#include <iostream>

using namespace std;
using namespace colombia_supremo;

int main(int argc, char **argv) {
  const uint32_t inputLength = 4;
  const uint32_t outputLength = 3;
  const auto weightShape = TensorShape({1, 1, inputLength, outputLength});
  TensorRawData weightData;
  {
    for (uint32_t r = 0; r < inputLength; ++r) {
      float v = 1.f;
      for (uint32_t c = 0; c < outputLength; ++c) {
        weightData.emplace_back(v++);
      }
    }
  }
  TensorRawData biasData({1.f, 2.f, 3.f});

  vector<shared_ptr<Operator>> operators;
  operators.emplace_back(make_shared<FullyConnectionOperator>(
      inputLength, outputLength,
      make_shared<WeightTensor>(weightData, weightShape),
      make_shared<TensorRawData>(biasData)));

  Session session(operators);

  TensorRawData input({1.f, 2.f, 3.f, 4.f});
  TensorRawData output({0.f, 0.f, 0.f});

  session.Run(input, output);

  for (const auto element : output) {
    cout << element << " ";
  }
  cout << endl;
  return 0;
}

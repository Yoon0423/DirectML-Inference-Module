#include <ConvolutionOperator.hpp>
#include <Session.hpp>
#include <iostream>
#include <random>

using namespace std;
using namespace colombia_supremo;

int main(int argc, char **argv) {
  random_device random;
  mt19937 generator(random());
  uniform_real_distribution<> distribution(-2.f, 2.f);

  const auto inputShape = TensorShape({1, 1, 5, 5});
  const auto outputShape = TensorShape({1, 3, 5, 5});
  const auto weightShape = TensorShape({3, 1, 3, 3});
  TensorRawData weightData;
  for (float v = 1.f; v <= 27.f; ++v) {
    weightData.emplace_back(distribution(generator));
  }

  const auto biasShape = TensorShape({1, 3, 1, 1});
  TensorRawData biasData;
  for (float v = 1.f; v <= 3.f; ++v) {
    biasData.emplace_back(distribution(generator));
  }

  vector<shared_ptr<Operator>> operators;
  operators.emplace_back(make_shared<ConvolutionOperator>(
      inputShape, outputShape,
      make_shared<WeightTensor>(weightData, weightShape),
      make_shared<WeightTensor>(biasData, biasShape), 1, 1, false));

  vector<shared_ptr<Operator>> operators_relu;
  operators_relu.emplace_back(make_shared<ConvolutionOperator>(
      inputShape, outputShape,
      make_shared<WeightTensor>(weightData, weightShape),
      make_shared<WeightTensor>(biasData, biasShape), 1, 1, true));

  TensorRawData input;
  input.reserve(25);
  for (float v = 1.f; v <= 25.f; v += 1.f) {
    input.emplace_back(distribution(generator));
  }

  TensorRawData output;
  output.reserve(75);
  for (size_t i = 0; i < 75; ++i) {
    output.emplace_back(0.f);
  }

  Session session(operators);
  session.Run(input, output);

  for (size_t c = 0; c < 3; ++c) {
    constexpr static size_t HW = 25;
    cout << "channel " << c + 1 << endl;
    for (size_t h = 0; h < 5; ++h) {
      constexpr static size_t W = 5;
      for (size_t w = 0; w < 5; ++w) {
        cout << output[HW * c + W * h + w] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  for (size_t i = 0; i < 75; ++i) {
    output[i] = 0.f;
  }

  Session session_relu(operators_relu);
  session_relu.Run(input, output);

  for (size_t c = 0; c < 3; ++c) {
    constexpr static size_t HW = 25;
    cout << "channel " << c + 1 << endl;
    for (size_t h = 0; h < 5; ++h) {
      constexpr static size_t W = 5;
      for (size_t w = 0; w < 5; ++w) {
        cout << output[HW * c + W * h + w] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  return 0;
}

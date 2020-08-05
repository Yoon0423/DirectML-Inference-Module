#include <ConvolutionOperator.hpp>
#include <iostream>

using namespace std;
using namespace colombia_supremo;

int main(int argc, char **argv) {
  const auto inputShape = TensorShape({1, 1, 5, 5});
  const auto outputShape = TensorShape({1, 3, 5, 5});
  const auto weightShape = TensorShape({3, 1, 3, 3});
  TensorRawData weightData;
  for (float v = 1.f; v <= 27.f; ++v) {
    weightData.emplace_back(v);
  }

  const auto biasShape = TensorShape({1, 3, 1, 1});
  TensorRawData biasData;
  for (float v = 1.f; v <= 3.f; ++v) {
    biasData.emplace_back(v);
  }

  const uint32_t stride = 1;
  const uint32_t padding = 1;

  const auto convOp = make_shared<ConvolutionOperator>(
      inputShape, outputShape,
      make_shared<WeightTensor>(weightData, weightShape),
      make_shared<WeightTensor>(biasData, biasShape), stride, padding);

  auto uploadTensor = convOp->CreateNewUploadTensor();
  TensorRawData input;
  for (float v = 1.f; v <= 25.f; ++v) {
    input.emplace_back(v);
  }
  uploadTensor->ReadFromData(input);

  auto readbackTensor = convOp->CreateNewReadbackTensor();

  convOp->Run(uploadTensor, readbackTensor);

  TensorRawData output;
  for (size_t i = 0; i < 75; ++i) {
    output.emplace_back(0.f);
  }

  readbackTensor->WriteToData(output);

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

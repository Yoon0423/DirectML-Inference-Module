#include <FullyConnectionOperator.hpp>
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

  auto fcOp = make_shared<FullyConnectionOperator>(
      inputLength, outputLength,
      make_shared<WeightTensor>(weightData, weightShape));

  auto uploadTensor = fcOp->CreateNewUploadTensor();
  uploadTensor->ReadFromData({-1, 2, -3, 4});

  auto readbackTensor = fcOp->CreateNewReadbackTensor();

  fcOp->Run(uploadTensor, readbackTensor);

  auto output = TensorRawData({0, 0, 0});

  readbackTensor->WriteToData(output);

  for (const auto element : output) {
    cout << element << " ";  
  }
  cout << endl;

  return 0;
}

// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include <MnistImageLoader.hpp>
#include <MnistLabelLoader.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <onnxruntime/onnxruntime_c_api.h>
#include <vector>

using namespace std;
using namespace chrono;
using namespace colombia_supremo;

void CheckStatus(OrtStatus *status);

const size_t width = 28;
const size_t inputTensorTotalBytes = width * width;
const size_t labelsCount = 10;
const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
const wchar_t *modelPath = L"../Model/mnist.onnx";
const vector<int64_t> inputDims = {1, 1, static_cast<int64_t>(width),
                                   static_cast<int64_t>(width)};
const vector<const char *> inputNodeNames = {"Input3"};
const vector<const char *> outputNodeNames = {"Plus214_Output_0"};

int main(int argc, char **argv) {
  cout << "TestOnnxruntime" << endl;

  OrtEnv *environment;
  CheckStatus(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mnist", &environment));

  OrtSessionOptions *sessionOptions;
  CheckStatus(api->CreateSessionOptions(&sessionOptions));
  CheckStatus(api->SetIntraOpNumThreads(sessionOptions, 0));
  CheckStatus(
      api->SetSessionGraphOptimizationLevel(sessionOptions, ORT_ENABLE_BASIC));

  OrtSession *session;
  CheckStatus(
      api->CreateSession(environment, modelPath, sessionOptions, &session));

  OrtAllocator *allocator;
  CheckStatus(api->GetAllocatorWithDefaultOptions(&allocator));

  OrtValue *inputTensor;
  CheckStatus(api->CreateTensorAsOrtValue(allocator, inputDims.data(), 4,
                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                          &inputTensor));
  int isTensor;
  CheckStatus(api->IsTensor(inputTensor, &isTensor));
  assert(isTensor);

  MnistImageLoader imageLoader("../Model/t10k-images.idx3-ubyte");
  MnistLabelLoader labelLoader("../Model/t10k-labels.idx1-ubyte");

  double elapsedNanoseconds = 0.;

  for (size_t i = 0; i < 100; ++i) {
    const auto imageData = imageLoader.LoadData(i);
    const auto label = labelLoader.LoadData(i);

    static float *inputData = nullptr;
    CheckStatus(api->GetTensorMutableData(
        inputTensor, reinterpret_cast<void **>(&inputData)));

    memcpy(reinterpret_cast<void *>(inputData),
           reinterpret_cast<const void *>(imageData.data()),
           inputTensorTotalBytes * sizeof(float));

    OrtValue *outputTensor = nullptr;

    const auto t_start = high_resolution_clock::now();
    CheckStatus(api->Run(session, NULL, inputNodeNames.data(),
                         (const OrtValue *const *)&inputTensor, 1,
                         outputNodeNames.data(), 1, &outputTensor));
    const auto t_end = high_resolution_clock::now();

    elapsedNanoseconds += static_cast<double>(duration_cast<nanoseconds>(t_end - t_start).count());

    CheckStatus(api->IsTensor(outputTensor, &isTensor));
    assert(isTensor);

    static float *outputData = nullptr;
    CheckStatus(api->GetTensorMutableData(
        outputTensor, reinterpret_cast<void **>(&outputData)));

    size_t result = 0;
    float value = outputData[0];
    for (size_t j = 0; j < 10; ++j) {
      if (outputData[j] > value) {
        result = j;
        value = outputData[j];
      }
    }

    //cout << "inferenced output: " << result
    //     << " | answer: " << static_cast<int>(label) << endl;

    api->ReleaseValue(outputTensor);
  }

  cout << "elapsed time: "
       << static_cast<long long>(elapsedNanoseconds / 1000000.0) << endl;

  api->ReleaseValue(inputTensor);
  api->ReleaseSession(session);
  api->ReleaseSessionOptions(sessionOptions);
  api->ReleaseEnv(environment);

  return 0;
}

void CheckStatus(OrtStatus *status) {
  if (status != NULL) {
    cout << api->GetErrorMessage(status) << endl;
    api->ReleaseStatus(status);
    exit(1);
  }
}

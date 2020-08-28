#include <FullyConnectionOperator.hpp>
#include <Session.hpp>
#include <chrono>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace std;
using namespace colombia_supremo;

int main(int argc, char **argv) {
  cout << "TestSession" << endl;

  random_device random;
  mt19937 generator(random());
  uniform_real_distribution<> distribution(-2.f, 2.f);

  const uint32_t inputLength = 4096;
  const uint32_t hiddenLength_1 = 1024;
  const uint32_t hiddenLength_2 = 128;
  const uint32_t outputLength = 10;

  TensorRawData inputs[3];
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < inputLength; ++j) {
      inputs[i].emplace_back(distribution(generator));
    }
  }

  TensorRawData outputsFromSingleThread[3];
  TensorRawData outputsFromMultiThread[3];

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < outputLength; ++j) {
      outputsFromSingleThread[i].emplace_back(0.f);
      outputsFromMultiThread[i].emplace_back(0.f);
    }
  }

  vector<shared_ptr<Operator>> operators;

  {
    const auto weightShape = TensorShape({1, 1, inputLength, hiddenLength_1});

    TensorRawData weightData;
    for (uint32_t r = 0; r < inputLength; ++r) {
      for (uint32_t c = 0; c < hiddenLength_1; ++c) {
        weightData.emplace_back(distribution(generator));
      }
    }

    operators.emplace_back(make_shared<FullyConnectionOperator>(
        inputLength, hiddenLength_1,
        make_shared<WeightTensor>(weightData, weightShape)));
  }

  {
    const auto weightShape =
        TensorShape({1, 1, hiddenLength_1, hiddenLength_2});

    TensorRawData weightData;
    for (uint32_t r = 0; r < hiddenLength_1; ++r) {
      for (uint32_t c = 0; c < hiddenLength_2; ++c) {
        weightData.emplace_back(distribution(generator));
      }
    }

    operators.emplace_back(make_shared<FullyConnectionOperator>(
        hiddenLength_1, hiddenLength_2,
        make_shared<WeightTensor>(weightData, weightShape)));
  }

  {
    const auto weightShape = TensorShape({1, 1, hiddenLength_2, outputLength});

    TensorRawData weightData;
    for (uint32_t r = 0; r < hiddenLength_2; ++r) {
      for (uint32_t c = 0; c < outputLength; ++c) {
        weightData.emplace_back(distribution(generator));
      }
    }

    operators.emplace_back(make_shared<FullyConnectionOperator>(
        hiddenLength_2, outputLength,
        make_shared<WeightTensor>(weightData, weightShape)));
  }

  {
    const auto t0 = chrono::high_resolution_clock::now();

    Session session(operators);

    for (size_t i = 0; i < 3; ++i) {
      session.Run(inputs[i], outputsFromSingleThread[i]);
    }

    const auto t1 = chrono::high_resolution_clock::now();

    cout << "Single thread: \n"
         << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count()
         << endl;

    for (size_t i = 0; i < 3; ++i) {
      auto output = outputsFromSingleThread[i];

      for (const auto element : output) {
        cout << element << " ";
      }

      cout << endl;
    }
  }

  {
    const auto t0 = chrono::high_resolution_clock::now();

    vector<Session> sessions;
    sessions.reserve(3);
    sessions.emplace_back(operators);
    sessions.emplace_back(operators);
    sessions.emplace_back(operators);

    const auto run = [](Session &session, TensorRawData &input,
                        TensorRawData &output, promise<bool> &promise) {
      session.Run(input, output);
      promise.set_value(true);
    };

    vector<future<bool>> futures;
    vector<promise<bool>> promises;
    promises.emplace_back(promise<bool>());
    promises.emplace_back(promise<bool>());
    promises.emplace_back(promise<bool>());
    thread threads[3];

    const size_t count = 3;

    for (size_t i = 0; i < count; ++i) {
      futures.emplace_back(move(promises[i].get_future()));

      threads[i] = thread(run, ref(sessions[i]), ref(inputs[i]),
                          ref(outputsFromMultiThread[i]), ref(promises[i]));
    }

    for (size_t i = 0; i < count; ++i) {
      futures[i].wait();
    }

    for (size_t i = 0; i < count; ++i) {
      threads[i].join();
    }

    const auto t1 = chrono::high_resolution_clock::now();

    cout << "Multi thread: \n"
         << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << endl;

    for (size_t i = 0; i < count; ++i) {
      auto output = outputsFromMultiThread[i];

      for (const auto element : output) {
        cout << element << " ";
      }
      cout << endl;
    }
  }

  return 0;
}

// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "MnistImageLoader.hpp"

#include <cstdlib>
#include <fstream>
#include <memory>

#if defined(_DEBUG)

#include <iostream>

#endif

namespace colombia_supremo {

MnistImageLoader::MnistImageLoader(const char *const filePath)
    : DataLoader<TensorRawData>(filePath), mData(std::vector<TensorRawData>()) {
  std::ifstream fileStream(mFilePath.c_str(), std::ios::binary | std::ios::in);

  if (fileStream.is_open() == false) {
#if defined(_DEBUG)
    std::cout << "invalid file path" << mFilePath << std::endl;
#endif
    abort();
  }

  fileStream.seekg(0, std::ios::beg);

  unsigned long magicNumber;
  fileStream.read(reinterpret_cast<char *>(&magicNumber),
                  sizeof(unsigned long));
  magicNumber = _byteswap_ulong(magicNumber);

  unsigned long dataCount;
  fileStream.read(reinterpret_cast<char *>(&dataCount), sizeof(unsigned long));
  dataCount = _byteswap_ulong(dataCount);

  unsigned long rows;
  fileStream.read(reinterpret_cast<char *>(&rows), sizeof(unsigned long));
  rows = _byteswap_ulong(rows);

  unsigned long cols;
  fileStream.read(reinterpret_cast<char *>(&cols), sizeof(unsigned long));
  cols = _byteswap_ulong(cols);

  if (magicNumber != 2051 || dataCount != 10000 || rows != 28 || cols != 28) {
#if defined(_DEBUG)
    std::cout << "invalid file data" << std::endl;
#endif
    abort();
  }

  mDataCount = static_cast<size_t>(dataCount);

  mShape = TensorShape(
      {1, 1, static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)});

  mData.reserve(static_cast<size_t>(dataCount));

  uint8_t value;
  for (size_t i = 0; i < static_cast<size_t>(dataCount); ++i) {
    TensorRawData data;
    data.reserve(static_cast<size_t>(rows * cols));
    for (size_t j = 0; j < static_cast<size_t>(rows * cols); ++j) {
      fileStream.read(reinterpret_cast<char *>(&value), sizeof(uint8_t));
      data.emplace_back(static_cast<float>(value));
    }

    mData.emplace_back(std::move(data));
  }

  fileStream.close();
}

TensorRawData MnistImageLoader::LoadData(const size_t index) {
  return mData[index];
}

} // namespace colombia_supremo

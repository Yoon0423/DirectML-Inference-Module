// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#include "MnistLabelLoader.hpp"

#include <cstdlib>
#include <fstream>
#include <memory>

#if defined(_DEBUG)

#include <iostream>

#endif

namespace colombia_supremo {

MnistLabelLoader::MnistLabelLoader(const char *const filePath)
    : DataLoader<uint8_t>(filePath), mData(std::vector<uint8_t>()) {
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

  if (magicNumber != 2049 || dataCount != 10000) {
#if defined(_DEBUG)
    std::cout << "invalid file data" << std::endl;
#endif
    abort();
  }

  mDataCount = static_cast<size_t>(dataCount);

  mData.reserve(static_cast<size_t>(dataCount));

  for (size_t i = 0; i < static_cast<size_t>(dataCount); ++i) {
    uint8_t value;
    fileStream.read(reinterpret_cast<char *>(&value), sizeof(uint8_t));

    mData.emplace_back(value);
  }

  fileStream.close();
}

uint8_t MnistLabelLoader::LoadData(const size_t index) { return mData[index]; }

} // namespace colombia_supremo

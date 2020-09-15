// Copyright (c) 2020 Yoonsung Kim. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace colombia_supremo {

template <typename T> class DataLoader {
public:
  DataLoader() = delete;
  DataLoader(const char *const filePath);
  ~DataLoader() = default;

  virtual T LoadData(const size_t index) = 0;
  size_t GetDataCount();

protected:
  std::string mFilePath;
  size_t mDataCount;
};

template <typename T>
DataLoader<T>::DataLoader(const char *const filePath)
    : mFilePath(std::move(std::string(filePath))), mDataCount(0) {}

template <typename T> size_t DataLoader<T>::GetDataCount() {
  return mDataCount;
}

} // namespace colombia_supremo

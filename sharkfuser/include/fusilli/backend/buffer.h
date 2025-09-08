// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the code to create and manage a Fusilli buffer
// which is an RAII wrapper around IREE HAL buffer for proper initialization,
// cleanup and lifetime management.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_BUFFER_H
#define FUSILLI_BACKEND_BUFFER_H

#include "fusilli/attributes/types.h"
#include "fusilli/backend/backend.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

#include <array>
#include <cstdint>

namespace fusilli {

template <typename DataType> class Buffer {
public:
  Buffer(const std::vector<int64_t> &shape, const std::vector<DataType> &data) {

  }

  // Delete copy constructors, keep default move constructor and destructor
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept = default;
  Buffer &operator=(Buffer &&) noexcept = default;
  ~Buffer() = default;

  // Allow Graph objects to access private Buffer methods
  // namely `getBufferView()`.
  friend class Graph;

private:
  // Returns a raw pointer to the underlying IREE runtime instance.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `FusilliHandle` objects and
  // only valid as long as at least one handle exists.
  iree_hal_buffer_view_t *getBufferView() const { return bufferView_.get(); }

  IreeHalBufferViewUniquePtrType bufferView_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_BUFFER_H

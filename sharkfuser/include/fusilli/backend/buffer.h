// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the code to create and manage a Fusilli buffer
// which is an RAII wrapper around IREE HAL buffer view for proper
// initialization, cleanup and lifetime management.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_BUFFER_H
#define FUSILLI_BACKEND_BUFFER_H

#include "fusilli/backend/backend.h"

#include <iree/runtime/api.h>

namespace fusilli {

class Buffer {
public:
  Buffer() {
    // Create a new IREE HAL buffer view.
    iree_hal_buffer_view_t *bufferView = nullptr;

    // Wrap the raw buffer_view ptr with a unique_ptr and custom deleter
    // for lifetime management.
    bufferView_ = IreeHalBufferViewUniquePtrType(bufferView);
  }

  // Returns a raw pointer to the underlying IREE HAL buffer view.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `Buffer` object and only valid
  // as long as this buffer exists.
  iree_hal_buffer_view_t *getBufferView() const { return bufferView_.get(); }

  // Delete copy constructors, keep default move constructor and destructor
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept = default;
  Buffer &operator=(Buffer &&) noexcept = default;
  ~Buffer() = default;

private:
  IreeHalBufferViewUniquePtrType bufferView_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_BUFFER_H

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
  Buffer() = default;

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
  explicit Buffer(IreeHalBufferViewUniquePtrType bufferView)
      : bufferView_(std::move(bufferView)) {}

  IreeHalBufferViewUniquePtrType bufferView_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_BUFFER_H

/*
class Buffer {
 public:
  // Type alias for unique_ptr with the custom deleter
  using BufferViewPtr = std::unique_ptr<iree_hal_buffer_view_t, BufferViewDeleter>;

  Buffer() = default;

  // Factory: Allocates a new buffer view and takes ownership
  static Buffer Allocate(...same args...) {
    iree_hal_buffer_view_t* raw_ptr = nullptr;
    iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
        ...args..., &raw_ptr);
    if (!iree_status_is_ok(status)) {
      throw std::runtime_error("Failed to allocate buffer view.");
    }
    return Buffer(BufferViewPtr(raw_ptr));
  }

  // Factory: Imports an existing view (retains + wraps)
  static Buffer Import(iree_hal_buffer_view_t* external_ptr) {
    if (!external_ptr) {
      throw std::invalid_argument("Cannot import null buffer view.");
    }
    iree_hal_buffer_view_retain(external_ptr);
    return Buffer(BufferViewPtr(external_ptr));
  }

  // Move constructor, move assignment: defaulted
  Buffer(Buffer&&) noexcept = default;
  Buffer& operator=(Buffer&&) noexcept = default;

  // No copy
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  // Accessor
  iree_hal_buffer_view_t* get() const { return buffer_view_.get(); }
  operator iree_hal_buffer_view_t*() const { return get(); }

  bool is_valid() const { return buffer_view_ != nullptr; }

 private:
  explicit Buffer(BufferViewPtr ptr) : buffer_view_(std::move(ptr)) {}

  BufferViewPtr buffer_view_;
};

*/
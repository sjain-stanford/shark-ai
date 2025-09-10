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
#include "fusilli/backend/handle.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

namespace fusilli {

class Buffer {
public:
  // Factory: Imports an existing view and retains ownership
  static ErrorOr<Buffer> import(iree_hal_buffer_view_t *externalBufferView) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Importing pre-allocated device buffer");
    FUSILLI_RETURN_ERROR_IF(externalBufferView == nullptr,
                            ErrorCode::RuntimeFailure,
                            "External buffer view is NULL");

    iree_hal_buffer_view_retain(externalBufferView);
    return ok(Buffer(IreeHalBufferViewUniquePtrType(externalBufferView)));
  }

  // Factory: Allocates a new buffer view and takes ownership
  template <typename T>
  static ErrorOr<Buffer>
  allocate(const FusilliHandle &handle,
           const std::vector<iree_hal_dim_t> &bufferShape,
           const std::vector<T> &bufferData) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Allocating new device buffer");

    iree_hal_buffer_view_t *rawBufferView = nullptr;
    FUSILLI_CHECK_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
        // IREE HAL device and allocator:
        handle.getDevice(), iree_hal_device_allocator(handle.getDevice()),
        // Shape rank and dimensions:
        bufferShape.size(), bufferShape.data(),
        // Element type:
        getIreeHalElementTypeForT<T>(),
        // Encoding type:
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            // Intended usage of this buffer (transfers, dispatches, etc):
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            // Access to allow to this memory:
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            // Where to allocate (host or device):
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        // The actual heap buffer to wrap or clone and its allocator:
        iree_make_const_byte_span(bufferData.data(),
                                  bufferData.size() * sizeof(T)),
        // Buffer view + storage are returned and owned by the caller:
        &rawBufferView));

    return ok(Buffer(IreeHalBufferViewUniquePtrType(rawBufferView)));
  }

  // Allow creating empty (nullptr) initialized Buffer which is
  // useful for creating placeholder output buffers that are
  // populated by IREE's destination passing style APIs such as
  // `iree_runtime_call_outputs_pop_front_buffer_view`. After
  // allocation of the raw `iree_hal_buffer_view_t *`, call
  // Buffer::reset to have RAII guarantees for owning/releasing it.
  Buffer() = default;

  // This is useful when starting with an empty Buffer (nullptr)
  // that is later populated with an allocated buffer view.
  void reset(iree_hal_buffer_view_t *newBufferView) noexcept {
    bufferView_.reset(newBufferView);
  }

  // Automatic (implicit) conversion operator for Buffer ->
  // iree_hal_buffer_view_t*
  operator iree_hal_buffer_view_t *() const { return getBufferView(); }

  // Delete copy constructors, keep default move constructor and destructor
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept = default;
  Buffer &operator=(Buffer &&) noexcept = default;
  ~Buffer() = default;

private:
  // Returns a raw pointer to the underlying IREE HAL buffer view.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `Buffer` object and only valid
  // as long as this buffer exists.
  iree_hal_buffer_view_t *getBufferView() const { return bufferView_.get(); }

  // Explicit constructor is private. Create `Buffer` using one of the
  // factory methods above - `Buffer::import` or `Buffer::allocate`.
  explicit Buffer(IreeHalBufferViewUniquePtrType bufferView)
      : bufferView_(std::move(bufferView)) {}

  IreeHalBufferViewUniquePtrType bufferView_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_BUFFER_H

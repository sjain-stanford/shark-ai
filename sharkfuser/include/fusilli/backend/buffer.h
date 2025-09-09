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
  static ErrorOr<Buffer> allocate(const FusilliHandle &handle,
                                  const std::vector<int64_t> &bufferShape,
                                  const std::vector<T> &bufferData) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Allocating new device buffer");

    std::vector<iree_hal_dim_t> bufferShapeCast(bufferShape.begin(),
                                                bufferShape.end());
    iree_hal_buffer_view_t *rawBufferView = nullptr;

    FUSILLI_CHECK_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
        // IREE HAL device and allocator:
        handle.getDevice(), iree_hal_device_allocator(handle.getDevice()),
        // Shape rank and dimensions:
        bufferShapeCast.size(), bufferShapeCast.data(),
        // Element type:
        // TODO: Configure based on T
        IREE_HAL_ELEMENT_TYPE_FLOAT_16,
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

  // Returns a raw pointer to the underlying IREE HAL buffer view.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `Buffer` object and only valid
  // as long as this buffer exists.
  iree_hal_buffer_view_t *getBufferView() const { return bufferView_.get(); }

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
  explicit Buffer(IreeHalBufferViewUniquePtrType bufferView)
      : bufferView_(std::move(bufferView)) {}

  IreeHalBufferViewUniquePtrType bufferView_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_BUFFER_H

// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the code to create and manage a Fusilli handle
// which is an RAII wrapper around shared IREE runtime resources
// (instances and devices) for proper initialization, cleanup and
// lifetime management.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_HANDLE_H
#define FUSILLI_BACKEND_HANDLE_H

#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/support/logging.h"

#include <iree/hal/buffer_view.h>
#include <iree/runtime/api.h>

namespace fusilli {

// An application using Fusilli to run operations on a given device
// must first initialize a handle on that device by calling
// `FusilliHandle::create()`. This allocates the necessary resources
// (runtime instance, HAL device) whose lifetimes are managed / owned
// by the handle(s).
//
// Here's a rough mapping of Fusilli constructs to IREE runtime constructs
// (based on scope and lifetime):
//
//  - Group of `FusilliHandle`s manage the IREE runtime instance lifetime.
//    An instance is shared across handles/threads/sessions and released
//    when the last handle goes out of scope.
//  - `FusilliHandle` manages IREE HAL device lifetime. Handles may be shared
//    by multiple graphs (as long as they intend to run on the same device).
//    Separate physical devices should have their own handles (hence logical
//    HAL device) created. Graphs running on the same physical devices should
//    reuse the same handle (hence logical HAL device). The device is released
//    when the handle holding it goes out of scope.
//  - `Graph` manages IREE runtime session lifetime. A session holds state on
//    the HAL device and the loaded VM modules.
class FusilliHandle {
public:
  static ErrorOr<FusilliHandle> create(Backend backend) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating handle for backend: " << backend);

    // Create a shared IREE runtime instance (thread-safe) and use it
    // along with the backend to construct a handle (without
    // initializing the device yet)
    auto handle = FusilliHandle(backend, FUSILLI_TRY(createSharedInstance()));

    // Lazy create handle-specific IREE HAL device and populate the handle
    FUSILLI_CHECK_ERROR(handle.createPerHandleDevice());

    return ok(std::move(handle));
  }

  template <typename T>
  ErrorOr<Buffer<T>> allocateBuffer(const std::vector<int64_t> &bufferShape,
                                    const std::vector<T> &bufferData) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Allocating device buffer");

    std::vector<iree_hal_dim_t> bufferShapeCast(bufferShape.begin(),
                                                bufferShape.end());

    Buffer<T> buffer = Buffer<T>();
    iree_hal_buffer_view_t *bufferView = buffer.getBufferView();

    iree_hal_allocator_t *device_allocator =
        iree_hal_device_allocator(device_.get());

    FUSILLI_CHECK_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
        device_.get(), device_allocator,
        // Shape rank and dimensions:
        bufferShapeCast.size(), bufferShapeCast.data(),
        // Element type:
        // TODO: Configure based on T
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
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
        &bufferView));

    return ok(std::move(buffer));
  }

  // Delete copy constructors, keep default move constructor and destructor
  FusilliHandle(const FusilliHandle &) = delete;
  FusilliHandle &operator=(const FusilliHandle &) = delete;
  FusilliHandle(FusilliHandle &&) noexcept = default;
  FusilliHandle &operator=(FusilliHandle &&) noexcept = default;
  ~FusilliHandle() = default;

  // Allow Graph objects to access private FusilliHandle methods
  // namely `getDevice()` and `getInstance()`.
  friend class Graph;

private:
  // Creates static singleton IREE runtime instance shared across
  // handles/threads
  static ErrorOr<IreeRuntimeInstanceSharedPtrType> createSharedInstance();

  // Creates IREE HAL device for this handle
  ErrorObject createPerHandleDevice();

  // Private constructor (use factory `create` method for handle creation)
  FusilliHandle(Backend backend, IreeRuntimeInstanceSharedPtrType instance)
      : backend_(backend), instance_(instance) {}

  Backend getBackend() const { return backend_; }

  // Returns a raw pointer to the underlying IREE HAL device.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `FusilliHandle` object and only
  // valid as long as this handle exists.
  iree_hal_device_t *getDevice() const { return device_.get(); }

  // Returns a raw pointer to the underlying IREE runtime instance.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `FusilliHandle` objects and only
  // valid as long as at least one handle exists.
  iree_runtime_instance_t *getInstance() const { return instance_.get(); }

  // Order of initialization matters here.
  // `device_` depends on `backend_` and `instance_`.
  Backend backend_;
  IreeRuntimeInstanceSharedPtrType instance_;
  IreeHalDeviceUniquePtrType device_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_HANDLE_H

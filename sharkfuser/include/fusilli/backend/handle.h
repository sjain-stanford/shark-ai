// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the code to create/destroy Fusilli handles.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_HANDLE_H
#define FUSILLI_BACKEND_HANDLE_H

#include "fusilli/backend/backend.h"
#include "fusilli/support/logging.h"

#include <iree/base/api.h>
#include <iree/hal/api.h>
#include <iree/runtime/api.h>

#include <memory>

namespace fusilli {

// Custom deleter for IREE runtime instance
struct IreeRuntimeInstanceDeleter {
  void operator()(iree_runtime_instance_t *instance) const {
    if (instance) {
      iree_runtime_instance_release(instance);
    }
  }
};

// Custom deleter for IREE HAL device
struct IreeHalDeviceDeleter {
  void operator()(iree_hal_device_t *device) const {
    if (device) {
      iree_hal_device_release(device);
    }
  }
};

class fusilliHandle {
public:
  fusilliHandle(Backend backend)
      : backend_(backend), instance_(getSharedInstance()),
        device_(getPerHandleDevice()) {}

  // No copies allowed, but moves are OK
  fusilliHandle(const fusilliHandle &) = delete;
  fusilliHandle &operator=(const fusilliHandle &) = delete;
  fusilliHandle(fusilliHandle &&) = default;
  fusilliHandle &operator=(fusilliHandle &&) = default;

  Backend getBackend() const { return backend_; }
  iree_hal_device_t *getDevice() const { return device_.get(); }
  iree_runtime_instance_t *getInstance() const { return instance_.get(); }

private:
  // Create IREE HAL device for this handle
  std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>
  getPerHandleDevice() {
    ErrorOr<std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>>
        uniqueDevice = [this]()
        -> ErrorOr<std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>> {
      iree_hal_device_t *rawDevice = nullptr;
      FUSILLI_CHECK_ERROR(iree_runtime_instance_try_create_default_device(
          instance_.get(), iree_make_cstring_view(halDriver.at(backend_)),
          &rawDevice));
      return ok(
          std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>(rawDevice));
    }();
    return std::move(*uniqueDevice);
  }

  // Create static singleton IREE runtime instance shared by all handles
  static std::shared_ptr<iree_runtime_instance_t> getSharedInstance() {
    static ErrorOr<std::shared_ptr<iree_runtime_instance_t>> sharedInstance =
        []() -> ErrorOr<std::shared_ptr<iree_runtime_instance_t>> {
      iree_runtime_instance_options_t opts;
      iree_runtime_instance_options_initialize(&opts);
      iree_runtime_instance_options_use_all_available_drivers(&opts);

      iree_runtime_instance_t *rawInstance = nullptr;
      FUSILLI_CHECK_ERROR(iree_runtime_instance_create(
          &opts, iree_allocator_system(), &rawInstance));

      return ok(std::shared_ptr<iree_runtime_instance_t>(
          rawInstance, IreeRuntimeInstanceDeleter()));
    }();
    return *sharedInstance;
  }

  Backend backend_;
  std::shared_ptr<iree_runtime_instance_t> instance_;
  std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter> device_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_HANDLE_H

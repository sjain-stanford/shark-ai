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
    iree_hal_device_t *raw_device = nullptr;
    iree_runtime_instance_try_create_default_device(
        instance_.get(), iree_make_cstring_view(halDriver.at(backend_)),
        &raw_device);
    return std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>(raw_device);
  }

  // Create static singleton IREE runtime instance shared by all handles
  static std::shared_ptr<iree_runtime_instance_t> getSharedInstance() {
    static std::shared_ptr<iree_runtime_instance_t> shared_instance = []() {
      iree_runtime_instance_options_t opts;
      iree_runtime_instance_options_initialize(&opts);
      iree_runtime_instance_options_use_all_available_drivers(&opts);

      iree_runtime_instance_t *raw_instance = nullptr;
      iree_runtime_instance_create(&opts, iree_allocator_system(),
                                   &raw_instance);

      return std::shared_ptr<iree_runtime_instance_t>(
          raw_instance, IreeRuntimeInstanceDeleter());
    }();
    return shared_instance;
  }

  Backend backend_;
  std::shared_ptr<iree_runtime_instance_t> instance_;
  std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter> device_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_HANDLE_H

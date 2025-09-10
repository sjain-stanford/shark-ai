// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains backend specific code like the `Backend` type, code to
// map from Backend to `iree-compile` flags, IREE runtime types and deleters.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_BACKEND_H
#define FUSILLI_BACKEND_BACKEND_H

#include "fusilli/attributes/types.h"

#include <iree/runtime/api.h>

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusilli {

// Target backend to run the generated kernels on
enum class Backend {
  CPU,
  GFX942,
};

static const std::unordered_map<Backend, std::string> BackendToStr = {
    {Backend::CPU, "CPU"},
    {Backend::GFX942, "GFX942"},
};

// Stream operator for Backend
inline std::ostream &operator<<(std::ostream &os, const Backend &backend) {
  auto it = BackendToStr.find(backend);
  if (it != BackendToStr.end())
    os << it->second;
  else
    os << "UNKNOWN_BACKEND";
  return os;
}

// Map from backend to IREE HAL driver name
static const std::unordered_map<Backend, const char *> halDriver = {
    {Backend::CPU, "local-task"},
    {Backend::GFX942, "hip"},
};

// Map from backend to IREE compile flags
static const std::unordered_map<Backend, std::vector<std::string>>
    backendFlags = {
        {
            Backend::CPU,
            {
                "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host",
            },
        },
        {
            Backend::GFX942,
            {
                "--iree-hal-target-backends=rocm",
                "--iree-hip-target=gfx942",
                "--iree-opt-level=O3",
            },
        },
};

// Map from native element type to IREE HAL element type
template <typename T>
struct IreeHalElementTypeTrait;

template <>
struct IreeHalElementTypeTrait<float> {
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
};

template <>
struct IreeHalElementTypeTrait<half> {  // assuming `half` is your fp16 type
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
};

template <>
struct IreeHalElementTypeTrait<int32_t> {
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_SINT_32;
};

// Add other specializations as needed...



// Custom deleter for IREE runtime instance
struct IreeRuntimeInstanceDeleter {
  void operator()(iree_runtime_instance_t *instance) const {
    if (instance)
      iree_runtime_instance_release(instance);
  }
};

// Custom deleter for IREE HAL device
struct IreeHalDeviceDeleter {
  void operator()(iree_hal_device_t *device) const {
    if (device)
      iree_hal_device_release(device);
  }
};

// Custom deleter for IREE runtime session
struct IreeRuntimeSessionDeleter {
  void operator()(iree_runtime_session_t *session) const {
    if (session)
      iree_runtime_session_release(session);
  }
};

// Custom deleter for IREE HAL buffer view
struct IreeHalBufferViewDeleter {
  void operator()(iree_hal_buffer_view_t *bufferView) const {
    if (bufferView)
      iree_hal_buffer_view_release(bufferView);
  }
};

// Aliases for IREE runtime types with custom deleters
using IreeRuntimeInstanceSharedPtrType =
    std::shared_ptr<iree_runtime_instance_t>;
using IreeHalDeviceUniquePtrType =
    std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>;
using IreeRuntimeSessionUniquePtrType =
    std::unique_ptr<iree_runtime_session_t, IreeRuntimeSessionDeleter>;
using IreeHalBufferViewUniquePtrType =
    std::unique_ptr<iree_hal_buffer_view_t, IreeHalBufferViewDeleter>;

} // namespace fusilli

#endif // FUSILLI_BACKEND_BACKEND_H

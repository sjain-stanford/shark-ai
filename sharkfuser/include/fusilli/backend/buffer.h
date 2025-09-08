// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the code to create and manage Fusilli handles
// which wrap around shared IREE runtime resources (instances and devices).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_BUFFER_H
#define FUSILLI_BACKEND_BUFFER_H

#include <iree/runtime/api.h>

#include <cstdint>

namespace fusilli {

template <typename DataType> struct Buffer {
  int64_t numElements = 0;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_BUFFER_H

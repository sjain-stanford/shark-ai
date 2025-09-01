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

#include "fusilli/attributes/types.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

namespace fusilli {

template <typename DataType> struct Buffer {};

} // namespace fusilli

#endif // FUSILLI_BACKEND_BUFFER_H

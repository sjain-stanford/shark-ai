# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


add_sharkfuser_test(
  NAME sharkfuser_amdgpu_tests
  SRCS
    test_graph.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

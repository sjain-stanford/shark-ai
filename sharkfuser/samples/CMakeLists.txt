# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Find prebuilt Catch2 library
find_package(Catch2 3 REQUIRED)

add_sharkfuser_sample(
  NAME sharkfuser_cpu_convolution_samples
  SRCS
    convolution/conv_fprop_cpu.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

# TODO: maybe combine cpu + gpu into one sample
add_sharkfuser_sample(
  NAME sharkfuser_gpu_convolution_samples
  SRCS
    convolution/conv_fprop_gpu.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

add_sharkfuser_sample(
  NAME sharkfuser_runtime_capi_samples
  SRCS
    runtime/runtime_api.c
  DEPS
    iree_runtime_unified
)

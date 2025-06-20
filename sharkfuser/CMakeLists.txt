# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.28)

# Avoid in-source builds (prevent build artifacts from cluttering the source directory)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(
    FATAL_ERROR
      "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source."
  )
endif()

# Get version number from file
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.json VERSION_JSON_STRING)
string(JSON PACKAGE_VERSION GET ${VERSION_JSON_STRING} package-version)
string(REGEX MATCH "(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*" BASE_VERSION ${PACKAGE_VERSION})

project(
  sharkfuser
  VERSION ${BASE_VERSION}
  DESCRIPTION "SharkFuser: A C++ Graph API backed by JIT compiled IREE code-generated kernels"
  LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory for header-only library
add_library(sharkfuser INTERFACE)
target_include_directories(sharkfuser INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Version pins for dependencies.
set(SHARKFUSER_IREE_GIT_TAG "iree-3.6.0rc20250612")

# build options
option(SHARKFUSER_BUILD_TESTS "Builds C++ tests" ON)


################################################################################
# Tests
################################################################################

if(SHARKFUSER_BUILD_TESTS)
  # Enable testing
  include(CTest)
  enable_testing()

  # Download Catch2 for testing
  if(NOT catch2_FOUND)
    include(FetchContent)

    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY  https://github.com/catchorg/Catch2.git
      GIT_TAG         v3.5.2
    )
    FetchContent_MakeAvailable(catch2)
  endif()

  # Add test executable
  add_executable(sharkfuser_tests
    tests/test_sharkfuser.cpp
  )
  target_link_libraries(sharkfuser_tests PRIVATE sharkfuser Catch2::Catch2WithMain)
  add_test(NAME sharkfuser_tests COMMAND sharkfuser_tests)

endif()

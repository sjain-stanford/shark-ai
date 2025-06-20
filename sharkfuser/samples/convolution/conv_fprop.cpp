// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sharkfuser.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Convolution fprop", "[conv][graph]") {
  namespace sf = sharkfuser;

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto build_new_graph = [=]() {};

  REQUIRE(sharkfuser::add(2, 3) == 5);
  REQUIRE(sharkfuser::add(-1, 1) == 0);
}

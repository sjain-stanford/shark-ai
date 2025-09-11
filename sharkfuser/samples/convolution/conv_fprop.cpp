// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "../../tests/utils.h"
#include "../utils.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <optional>

using namespace fusilli;

TEST_CASE("Convolution fprop", "[conv][graph]") {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto build_new_graph = [=](const FusilliHandle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("fprop_sample");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

    auto X = graph->tensor(TensorAttr()
                               .setName("image")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, h * w, w, 1}));

    auto W = graph->tensor(TensorAttr()
                               .setName("filter")
                               .setDim({k, c, r, s})
                               .setStride({c * r * s, r * s, s, 1}));

    auto conv_attr = ConvFPropAttr()
                         .setPadding({0, 0})
                         .setStride({1, 1})
                         .setDilation({1, 1})
                         .setName("conv_fprop");

    auto Y = graph->convFProp(X, W, conv_attr);

    // Specify Y's dimensions and strides
    Y->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
    Y->setOutput(true);

    REQUIRE(isOk(graph->validate()));

    REQUIRE(isOk(graph->compile(handle, /*remove=*/true)));

    return std::make_tuple(graph, X, W, Y);
  };

  // Parameterize sample by backend and create device-specific handles
  std::optional<ErrorOr<FusilliHandle>> handleOrError;
  SECTION("cpu backend") {
    handleOrError.emplace(FusilliHandle::create(Backend::CPU));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("gfx942 backend") {
    handleOrError.emplace(FusilliHandle::create(Backend::GFX942));
  }
#endif
  REQUIRE(handleOrError.has_value());
  REQUIRE(isOk(*handleOrError));
  FusilliHandle &handle = **handleOrError;

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, X, W, Y] = build_new_graph(handle);

  // Populate and check input buffer
  auto xBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT({n, c, h, w}),
                       /*data=*/std::vector<half>(n * c * h * w, half(1.0f)))));
  REQUIRE(*xBuf != nullptr);
  std::vector<half> input;
  REQUIRE(isOk(xBuf->read(handle, input)));
  for (auto val : input) {
    REQUIRE(val == half(1.0f));
  }

  // Populate and check weight buffer
  auto wBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT({k, c, r, s}),
                       /*data=*/std::vector<half>(k * c * r * s, half(1.0f)))));
  REQUIRE(*wBuf != nullptr);
  std::vector<half> weight;
  REQUIRE(isOk(wBuf->read(handle, weight)));
  for (auto val : weight) {
    REQUIRE(val == half(1.0f));
  }

  // Create empty result buffer
  auto yBuf = std::make_shared<Buffer>();
  REQUIRE(*yBuf == nullptr);

  // Create variant pack
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };

  // Execute graph
  REQUIRE(isOk(graph->execute(variantPack)));

  // Check result buffer
  REQUIRE(*yBuf != nullptr);
  std::vector<half> result;
  REQUIRE(isOk(yBuf->read(handle, result)));
  for (auto val : result) {
    REQUIRE(val == half(128.0f));
  }
}

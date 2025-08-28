// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "../utils.h"

#include <catch2/catch_test_macros.hpp>
#include <optional>

using namespace fusilli;

// Helper function to create a valid graph for testing
Graph validGraph() {
  Graph g;
  g.setName("test_graph");
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;
  g.setName("test_graph");
  g.setIODataType(DataType::Half).setComputeDataType(DataType::Float);
  auto X = g.tensor(TensorAttr()
                        .setName("image")
                        .setDim({n, c, h, w})
                        .setStride({c * h * w, h * w, w, 1}));
  auto W = g.tensor(TensorAttr()
                        .setName("filter")
                        .setDim({k, c, r, s})
                        .setStride({c * r * s, r * s, s, 1}));
  auto conv = ConvFPropAttr()
                  .setPadding({0, 0})
                  .setStride({1, 1})
                  .setDilation({1, 1})
                  .setName("conv_fprop");
  auto Y = g.convFProp(X, W, conv);
  Y->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
  Y->setOutput(true);
  REQUIRE(isOk(g.validate()));
  return g;
};

TEST_CASE("Graph `readOrGenerateCompiledArtifact`", "[graph]") {
  SECTION("cache generation and invalidation") {
    FusilliHandle cpuHandle =
        FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::CPU));
    FusilliHandle gpuHandle =
        FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::GFX942));

    Graph g = validGraph();

    std::string generatedAsm = FUSILLI_REQUIRE_UNWRAP(g.emitAsm());

    // Cache should be empty, compilation artifacts should be generated.
    std::optional<bool> reCompiled = std::nullopt;
    REQUIRE(isOk(g.readOrGenerateCompiledArtifact(cpuHandle, generatedAsm,
                                                  /*remove=*/true,
                                                  /*reCompiled=*/&reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(reCompiled.value());

    // Cache should hit, no compilation should be required.
    reCompiled = std::nullopt;
    REQUIRE(isOk(g.readOrGenerateCompiledArtifact(cpuHandle, generatedAsm,
                                                  /*remove=*/true,
                                                  /*reCompiled=*/&reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(!reCompiled.value());

    // Cache should miss based on different handle / device / compile command.
    reCompiled = std::nullopt;
    REQUIRE(isOk(g.readOrGenerateCompiledArtifact(gpuHandle, generatedAsm,
                                                  /*remove=*/true,
                                                  /*reCompiled=*/&reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(reCompiled.value());

    // Cache should hit with the different handle the second time.
    reCompiled = std::nullopt;
    REQUIRE(isOk(g.readOrGenerateCompiledArtifact(gpuHandle, generatedAsm,
                                                  /*remove=*/true,
                                                  /*reCompiled=*/&reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(!reCompiled.value());
  }
}

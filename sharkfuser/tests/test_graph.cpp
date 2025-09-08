// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using namespace fusilli;

TEST_CASE("Graph getName correctly propagates the context name", "[graph]") {
  Graph g;
  g.setName("foo_graph");
  REQUIRE(g.getName() == "foo_graph");
}

TEST_CASE("Graph tensor() adds input tensor", "[graph]") {
  Graph g;
  g.setName("adds_input_tensor");
  auto t =
      g.tensor(TensorAttr().setName("input").setDim({2, 2}).setStride({2, 1}));
  REQUIRE(t->getName() == "input");
  REQUIRE(t->getDim() == std::vector<int64_t>({2, 2}));
  REQUIRE(t->getStride() == std::vector<int64_t>({2, 1}));
}

TEST_CASE("Graph conv_fprop() adds ConvFPropNode and output tensor",
          "[graph]") {
  Graph g;
  g.setName("adds_convfpropnode_and_output_tensor");
  auto x =
      g.tensor(TensorAttr().setDim({1, 8, 8, 3}).setStride({192, 24, 3, 1}));
  auto w = g.tensor(TensorAttr().setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
  auto y = g.convFProp(x, w, attr);

  // Names for inputs are auto-populated when not set
  REQUIRE(x->getName() == "conv_fprop_0_X");
  REQUIRE(w->getName() == "conv_fprop_0_W");
  REQUIRE(y->getName() == "conv_fprop_0_Y");

  // Y is virtual (intermediate tensor) unless specified as output
  REQUIRE(y->isVirtual() == true);
  y->setOutput(true);
  REQUIRE(y->isVirtual() == false);
}

TEST_CASE("Graph validate() fails if name is not set", "[graph]") {
  Graph g;
  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
  REQUIRE(status.getMessage() == "Graph name not set");

  g.setName("name_is_set_now");
  REQUIRE(isOk(g.validate()));
}

TEST_CASE("Graph validate() fails on missing attributes", "[graph]") {
  Graph g;
  g.setName("validate_returns_ok_for_valid_graph");
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 8, 8, 3})
                        .setStride({192, 24, 3, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1}).setName("conv");
  auto y = g.convFProp(x, w, attr);

  // Fails because y is underspecified (shape/stride inference unimplemented)
  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  REQUIRE(status.getMessage() ==
          "ConvFProp node shape inference not implemented yet; please "
          "specify output tensor dimensions");

  // Specify y's shape and strides
  y->setDim({1, 8, 8, 4}).setStride({256, 32, 4, 1});
  REQUIRE(isOk(g.validate()));
}

// Helper function to create graph for testing
Graph testGraph(bool validate) {
  Graph g;
  g.setName("unvalidated_graph");
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;
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
  if (validate) {
    g.setName("validated_graph");
    REQUIRE(isOk(g.validate()));
  }
  return g;
};

TEST_CASE("Graph asm_emitter requires validation to be run first", "[graph]") {
  Graph g = testGraph(/*validate=*/false);

  // ASM emitter without validation should throw an error
  auto status = g.emitAsm();
  REQUIRE(isError(status));
  REQUIRE(ErrorObject(status).getCode() == ErrorCode::NotValidated);
  REQUIRE(ErrorObject(status).getMessage() ==
          "Graph must be validated before emitting MLIR assembly");

  // Validate the graph first
  REQUIRE(isOk(g.validate()));

  // ASM emitter should now work
  REQUIRE(isOk(g.emitAsm()));
}

TEST_CASE("Graph `getCompiledArtifact` cache generation and invalidation",
          "[graph]") {
  FusilliHandle cpuHandle =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::CPU));
#ifdef FUSILLI_ENABLE_AMDGPU
  FusilliHandle gpuHandle =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::GFX942));
#endif

  Graph g = testGraph(/*validate=*/true);

  std::string generatedAsm = FUSILLI_REQUIRE_UNWRAP(g.emitAsm());

  // Cache should be empty, compilation artifacts should be generated.
  std::optional<bool> reCompiled = std::nullopt;
  REQUIRE(isOk(g.getCompiledArtifact(cpuHandle, generatedAsm, /*remove=*/true,
                                     &reCompiled)));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());

  // Cache should hit, no compilation should be required.
  reCompiled = std::nullopt;
  REQUIRE(isOk(g.getCompiledArtifact(cpuHandle, generatedAsm, /*remove=*/true,
                                     &reCompiled)));
  REQUIRE(reCompiled.has_value());
  REQUIRE(!reCompiled.value());

#ifdef FUSILLI_ENABLE_AMDGPU
  // Cache should miss based on different handle / device / compile command.
  reCompiled = std::nullopt;
  REQUIRE(isOk(g.getCompiledArtifact(gpuHandle, generatedAsm, /*remove=*/true,
                                     &reCompiled)));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());

  // Cache should hit with a re-run on the different handle.
  reCompiled = std::nullopt;
  REQUIRE(isOk(g.getCompiledArtifact(gpuHandle, generatedAsm, /*remove=*/true,
                                     &reCompiled)));
  REQUIRE(reCompiled.has_value());
  REQUIRE(!reCompiled.value());
#endif

  // Cache should miss because of different generated asm.
  reCompiled = std::nullopt;
  REQUIRE(isOk(g.getCompiledArtifact(cpuHandle, generatedAsm + " ",
                                     /*remove=*/true, &reCompiled)));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());

  // Cache should hit with the same generated asm.
  reCompiled = std::nullopt;
  REQUIRE(isOk(g.getCompiledArtifact(cpuHandle, generatedAsm + " ",
                                     /*remove=*/true, &reCompiled)));
  REQUIRE(reCompiled.has_value());
  REQUIRE(!reCompiled.value());

  // Cache should miss because graph name change.
  g.setName("new_graph_name");
  reCompiled = std::nullopt;
  REQUIRE(isOk(g.getCompiledArtifact(cpuHandle, generatedAsm + " ",
                                     /*remove=*/true, &reCompiled)));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());
}

TEST_CASE("Graph `getCompiledArtifact` should not read cached items from "
          "other/previous Graph instances",
          "[graph]") {
  FusilliHandle handle =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::CPU));

  std::string generatedAsm;
  {
    Graph g = testGraph(/*validate=*/true);

    generatedAsm = FUSILLI_REQUIRE_UNWRAP(g.emitAsm());

    // Cache should be empty.
    std::optional<bool> reCompiled = std::nullopt;
    REQUIRE(isOk(g.getCompiledArtifact(handle, generatedAsm, /*remove=*/false,
                                       &reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(reCompiled.value());

    // Cache should hit with the same generated asm.
    reCompiled = std::nullopt;
    REQUIRE(isOk(g.getCompiledArtifact(handle, generatedAsm, /*remove=*/false,
                                       &reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(!reCompiled.value());
  }

  Graph g = testGraph(/*validate=*/true);

  // Check that the generated asm matches the cache.
  CacheFile asmCache = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::open(g.getName(), IREE_COMPILE_INPUT_FILENAME));
  REQUIRE(FUSILLI_REQUIRE_UNWRAP(asmCache.read()) == generatedAsm);

  // Nonetheless a new instance should regenerate cache.
  std::optional<bool> reCompiled = std::nullopt;
  REQUIRE(isOk(g.getCompiledArtifact(handle, generatedAsm, /*remove=*/true,
                                     &reCompiled)));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());
}

TEST_CASE("Graph `getCompiledArtifact` invalid input IR", "[graph]") {
  FusilliHandle handle =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::CPU));
  std::string graphName;
  {
    Graph g;
    g.setName("invalid_input_ir");
    ErrorObject err =
        g.getCompiledArtifact(handle, "invalid mlir", /*remove=*/true);
    REQUIRE(isError(err));
    REQUIRE(err.getCode() == ErrorCode::CompileFailure);
    REQUIRE(err.getMessage() == "iree-compile command failed");
  }
  // Cache created with "remove", ensure it is removed after the test.
  REQUIRE(!std::filesystem::exists(
      CacheFile::getPath(graphName, "test").parent_path()));
}

TEST_CASE("Graph `compile` method fails without validation", "[graph]") {
  FusilliHandle handle =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::CPU));

  Graph g = testGraph(/*validate=*/false);

  auto status = g.compile(handle, /*remove=*/true);
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotValidated);
  REQUIRE(status.getMessage() ==
          "Graph must be validated before being compiled");
}

TEST_CASE("Graph `compile` recompilations with changed handle", "[graph]") {
  // This test constructs a single graph but compiles it with different
  // handles and backends, ensuring that the graph did not use cached
  // artifacts from a previous compilation and correctly re-compiled
  // for the new handle/backend.
  Graph g = testGraph(/*validate=*/true);

  // Path to compile command cache file
  const char *cacheDir = std::getenv("FUSILLI_CACHE_DIR");
  if (!cacheDir)
    cacheDir = std::getenv("HOME");
  std::filesystem::path cmdPath = std::filesystem::path(cacheDir) / ".cache" /
                                  "fusilli" / g.getName() /
                                  "iree-compile-command.txt";

  FusilliHandle cpuHandle =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::CPU));
  REQUIRE(isOk(g.compile(cpuHandle, /*remove=*/true)));

  std::string cpuCmd;
  REQUIRE(std::filesystem::exists(cmdPath));
  std::ifstream cpuCmdFile(cmdPath);
  REQUIRE(cpuCmdFile.is_open());
  std::getline(cpuCmdFile, cpuCmd);
  REQUIRE(!cpuCmd.empty());

#ifdef FUSILLI_ENABLE_AMDGPU
  FusilliHandle gpuHandle =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::GFX942));
  REQUIRE(isOk(g.compile(gpuHandle, /*remove=*/true)));

  std::string gpuCmd;
  REQUIRE(std::filesystem::exists(cmdPath));
  std::ifstream gpuCmdFile(cmdPath);
  REQUIRE(gpuCmdFile.is_open());
  std::getline(gpuCmdFile, gpuCmd);
  REQUIRE(!gpuCmd.empty());

  // The compile commands should be different for CPU and GPU handles
  REQUIRE(cpuCmd != gpuCmd);
#endif
}

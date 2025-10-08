// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "fusilli/backend/backend.h"
#include "fusilli/support/logging.h"
#include "utils.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

ErrorObject benchmark_conv_fprop() {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;
  Handle handle = FUSILLI_TRY(Handle::create(Backend::GFX942));

  auto graph = std::make_shared<Graph>();
  graph->setName("conv_fprop_sample_nchw_kcrs_1x1_nopad");
  graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

  auto X = graph->tensor(TensorAttr()
                             .setName("image")
                             .setDim({n, c, h, w})
                             .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto W = graph->tensor(TensorAttr()
                             .setName("filter")
                             .setDim({k, c, r, s})
                             .setStride({c * r * s, r * s, s, 1})); // KCRS

  auto conv_attr = ConvFPropAttr()
                       .setPadding({0, 0})
                       .setStride({1, 1})
                       .setDilation({1, 1})
                       .setName("conv_fprop");

  auto Y = graph->convFProp(X, W, conv_attr);
  Y->setOutput(true);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph->validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));

  // Allocate input buffer.
  auto xBuf = std::make_shared<Buffer>(FUSILLI_TRY(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT({n, c, h, w}),
                       /*data=*/std::vector<half>(n * c * h * w, half(1.0f)))));

  // Allocate weight buffer.
  auto wBuf = std::make_shared<Buffer>(FUSILLI_TRY(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT({k, c, r, s}),
                       /*data=*/std::vector<half>(k * c * r * s, half(1.0f)))));

  // Allocate output buffer.
  auto yBuf = std::make_shared<Buffer>(FUSILLI_TRY(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT({n, k, h, w}),
                       /*data=*/std::vector<half>(n * k * h * w, half(0.0f)))));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };

  // Execute graph once.
  FUSILLI_CHECK_ERROR(graph->execute(variantPack));

  // Execute graph a few times.
  for (size_t i = 0; i < 5; i++)
    FUSILLI_CHECK_ERROR(graph->execute(variantPack));

  return ok();
}

int main() {
  auto status = benchmark_conv_fprop();
  std::cout << "Fusilli Benchmark complete!" << std::endl;
  return 0;
}

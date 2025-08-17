// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <iree/base/api.h>
#include <iree/base/config.h>
#include <iree/base/status.h>
#include <iree/base/string_view.h>
#include <iree/runtime/call.h>
#include <iree/runtime/instance.h>
#include <iree/runtime/session.h>
#include <iree/tooling/device_util.h>
#include <iree/vm/list.h>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// TODO: remove >>>>>>>>>>>>>>>>>>>>>>>
#define FUSILLI_REQUIRE_UNWRAP(expr)                                           \
  ({                                                                           \
    auto _errorOr = (expr);                                                    \
    REQUIRE(isOk(_errorOr));                                                   \
    std::move(*_errorOr);                                                      \
  })
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

using namespace fusilli;

TEST_CASE("Convolution fprop", "[conv][graph]") {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto graph = std::make_shared<Graph>();

  // Parameterize sample by backend
  SECTION("CPU backend") { graph->setBackend(Backend::CPU); }
  SECTION("gfx942 backend") { graph->setBackend(Backend::GFX942); }

  graph->setName("fprop_sample");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

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

  // ----------------------------------------------------------------------
  //  Create %filter + %image args
  // ----------------------------------------------------------------------

  // Get allocators
  iree_hal_device_t *device = FUSILLI_REQUIRE_UNWRAP(graph->getDevice());
  iree_hal_allocator_t *device_allocator =
      iree_hal_device_allocator(FUSILLI_REQUIRE_UNWRAP(graph->getDevice()));
  iree_allocator_t host_allocator = iree_allocator_system();

  // %filter: !torch.vtensor<[256,128,1,1],f32>
  iree_hal_buffer_view_t *filter = nullptr;
  {
    // Create hall buffer view with copy of the data.
    const std::array filterShape =
        std::to_array<iree_host_size_t>({256, 128, 1, 1});
    // The data should be copied, so lifetime of filterData isn't a problem
    // here. If we started using
    // iree_hal_buffer_view_generate_buffer_in_situ`
    // we would definitely have a use after free here.
    size_t size = std::accumulate(filterShape.begin(), filterShape.end(), 1,
                                  std::multiplies<size_t>());
    std::vector<float> filterData(size, 1.0f);
    REQUIRE(isOk(iree_hal_buffer_view_allocate_buffer_copy(
        device, device_allocator,
        // Shape rank and dimensions:
        filterShape.size(), filterShape.data(),
        // Element type:
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        // Encoding type:
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            // Intended usage of the buffer (transfers, dispatches, etc):
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            // Access to allow to this memory:
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            // Where to allocate (host or device):
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        // The actual heap buffer to wrap or clone and its allocator:
        iree_make_const_byte_span(filterData.data(),
                                  filterData.size() * sizeof(float)),
        // Buffer view + storage are returned and owned by the caller:
        &filter)));
  }

  // %image: !torch.vtensor<[16,128,64,64],f32>
  iree_hal_buffer_view_t *image = nullptr;
  {
    // Create hall buffer view with copy of the data.
    const std::array imageShape =
        std::to_array<iree_host_size_t>({16, 128, 64, 64});
    // The data should be copied, so lifetime of filterData isn't a problem
    // here. If we started using
    // iree_hal_buffer_view_generate_buffer_in_situ`
    // we would definitely have a use after free here.
    size_t size = std::accumulate(imageShape.begin(), imageShape.end(), 1,
                                  std::multiplies<size_t>());
    std::vector<float> imageData(size, 1.0f);
    REQUIRE(isOk(iree_hal_buffer_view_allocate_buffer_copy(
        device, device_allocator,
        // Shape rank and dimensions:
        imageShape.size(), imageShape.data(),
        // Element type:
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        // Encoding type:
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            // Intended usage of the buffer (transfers, dispatches, etc):
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            // Access to allow to this memory:
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            // Where to allocate (host or device):
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        // The actual heap buffer to wrap or clone and its allocator:
        iree_make_const_byte_span(imageData.data(),
                                  imageData.size() * sizeof(float)),
        // Buffer view + storage are returned and owned by the caller:
        &image)));
  }

  // ----------------------------------------------------------------------
  // graph->execute and check the results
  // ----------------------------------------------------------------------

  iree_runtime_call_t call = FUSILLI_REQUIRE_UNWRAP(
      graph->execute(std::vector<iree_hal_buffer_view_t *>{filter, image}));

  // Pull the results from the call.
  iree_hal_buffer_view_t *ret0 = nullptr;
  REQUIRE(isOk(iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret0)));

  // Copy results back from device (this also works for CPUs).
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(ret0);
  iree_device_size_t byte_length = iree_hal_buffer_view_byte_length(ret0);
  std::vector<float> hostData(byte_length / sizeof(float));
  REQUIRE(isOk(iree_hal_device_transfer_d2h(
      device, buffer, 0, hostData.data(), byte_length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout())));

  // Check the results.
  for (auto v : hostData) {
    REQUIRE(v == 128.0f);
  }

  // Under the current API it's hte callers responsibility to clean up the call.
  iree_runtime_call_deinitialize(&call);
}

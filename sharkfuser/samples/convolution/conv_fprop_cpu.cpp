// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fusilli/backend/backend.h"
#include "fusilli/support/logging.h"
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
#include <iree/hal/buffer.h>
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

TEST_CASE("Convolution fprop CPU", "[conv][graph]") {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto graph = std::make_shared<Graph>();
  graph->setBackend(Backend::CPU);
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

  ErrorOr<std::string> generatedAsm = graph->emitAsm();
  REQUIRE(isOk(generatedAsm));

  std::string ouputPath = FUSILLI_REQUIRE_UNWRAP(
      graph->readOrGenerateCompiledArtifact(*generatedAsm));

  // ----------------------------------------------------------------------
  //  Create instance + session
  // ----------------------------------------------------------------------

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t *instance = NULL;
  REQUIRE(isOk(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance)));

  // Grab the default device for "local-task" driver. Where "local-task" is
  // synchronous CPU.
  iree_hal_device_t *device = NULL;
  REQUIRE(isOk(iree_runtime_instance_try_create_default_device(
      instance,
      iree_make_cstring_view(halDriver.at(graph->context.getBackend())),
      &device)));

  // Create session
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t *session = NULL;
  REQUIRE(isOk(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session)));
  iree_hal_device_release(device);

  // ----------------------------------------------------------------------
  //  Load module + create call
  // ----------------------------------------------------------------------

  // append our file to the session
  REQUIRE(isOk(iree_runtime_session_append_bytecode_module_from_file(
      session, ouputPath.c_str())));

  // Initialize the call to the function.
  iree_runtime_call_t call;
  REQUIRE(isOk(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.main"), &call)));

  iree_vm_function_t function;
  REQUIRE(isOk(iree_runtime_session_lookup_function(
      session, iree_make_cstring_view("module.main"), &function)));

  // ----------------------------------------------------------------------
  //  Create %filter + %image args and add to the call
  // ----------------------------------------------------------------------

  // Get allocators
  iree_hal_allocator_t *device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);

  {
    // %filter: !torch.vtensor<[256,128,1,1],f32>
    iree_hal_buffer_view_t *filter = nullptr;

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

    // add the filter to the call inputs list
    REQUIRE(
        isOk(iree_runtime_call_inputs_push_back_buffer_view(&call, filter)));

    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(filter);
  }

  {
    // %image: !torch.vtensor<[16,128,64,64],f32>
    iree_hal_buffer_view_t *image = nullptr;

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

    // add the image to the call inputs list
    REQUIRE(isOk(iree_runtime_call_inputs_push_back_buffer_view(&call, image)));

    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(image);
  }

  FUSILLI_LOG_LABEL_ENDL("iree_vm_list_size(iree_runtime_call_inputs(&call)): "
                         << iree_vm_list_size(iree_runtime_call_inputs(&call)));

  // ----------------------------------------------------------------------
  // invoke the call and print the results
  // ----------------------------------------------------------------------

  REQUIRE(isOk(iree_runtime_call_invoke(&call, /*flags=*/0)));

  // Dump the function outputs.
  iree_hal_buffer_view_t *ret0 = NULL;
  // Try to get the first call result as a buffer view.
  REQUIRE(isOk(iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret0)));

  //   iree_hal_buffer_view_fprint(
  //         stdout, ret0, /*max_element_count=*/4096, host_allocator);

  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(ret0);
  iree_device_size_t byte_length = iree_hal_buffer_view_byte_length(ret0);
  std::vector<float> hostData(byte_length / sizeof(float));

  iree_hal_buffer_mapping_t buffer_mapping = {{0}};
  iree_status_t status = iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(ret0), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, &buffer_mapping);

  for (size_t i = 0; i < buffer_mapping.contents.data_length / sizeof(float);
       i++) {
    REQUIRE(((float *)buffer_mapping.contents.data)[i] == 128.0f);
  }

  //   // Check the result
  //   for (auto v : hostData) {
  //     REQUIRE(v == 128.0f);
  //   }

  // ----------------------------------------------------------------------
  //  cleanup
  // ----------------------------------------------------------------------

  iree_runtime_call_deinitialize(&call);
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);

  FUSILLI_LOG_LABEL_ENDL(FUSILLI_COLOR_GREEN
                         << "Runtime integration working so far!"
                         << FUSILLI_COLOR_RESET);
}

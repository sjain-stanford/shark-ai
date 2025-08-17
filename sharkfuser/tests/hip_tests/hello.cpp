#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <iree/base/api.h>
#include <iree/base/config.h>
#include <iree/base/status.h>
#include <iree/base/string_view.h>
#include <iree/hal/allocator.h>
#include <iree/runtime/call.h>
#include <iree/runtime/instance.h>
#include <iree/runtime/session.h>
#include <iree/tooling/device_util.h>
#include <iree/vm/list.h>

using namespace fusilli;

__global__ void hello_kernel() {
  printf("Hello from GPU! block %d thread %d\n", blockIdx.x, threadIdx.x);
}

TEST_CASE("proof of life for HIP", "[hip_tests]") {
  // ----------------------------------------------------------------------
  //  proof of life for GPU connection
  // ----------------------------------------------------------------------

  int dev = 0;
  hipDeviceProp_t prop{};
  hipGetDevice(&dev);
  hipGetDeviceProperties(&prop, dev);

  void *ptr;
  hipMalloc(&ptr, sizeof(float) * 64);

  // Launch kernel (1 block, 4 threads)
  hipLaunchKernelGGL(hello_kernel, dim3(1), dim3(4), 0, 0);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr, "hipDeviceSynchronize: %s\n", hipGetErrorString(err));
  }
  REQUIRE(err == hipSuccess);
}

// TODO: remove >>>>>>>>>>>>>>>>>>>>>>>
#define FUSILLI_REQUIRE_UNWRAP(expr)                                           \
  ({                                                                           \
    auto _errorOr = (expr);                                                    \
    REQUIRE(isOk(_errorOr));                                                   \
    std::move(*_errorOr);                                                      \
  })
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

TEST_CASE("Buffer import", "[hip_tests]") {

  Graph graph;
  graph.setBackend(Backend::GFX942);

  iree_hal_allocator_t *device_allocator =
      iree_hal_device_allocator(FUSILLI_REQUIRE_UNWRAP(graph.getDevice()));

  void *ptr;
  hipMalloc(&ptr, sizeof(float) * 64);
  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      .flags = 0,
      .size = sizeof(float) * 64,
      .handle =
          {
              .device_allocation =
                  {
                      .ptr = (uint64_t)ptr,
                  },
          },
  };

  iree_hal_buffer_params_t buffer_params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
  };
  iree_allocator_t system_allocator = iree_allocator_system();
  iree_hal_buffer_t *imported_buffer = NULL;
  iree_hal_buffer_release_callback_t release_callback =
      iree_hal_buffer_release_callback_null();
  REQUIRE(isOk(iree_hal_allocator_import_buffer(
      device_allocator, buffer_params, &external_buffer, release_callback,
      &imported_buffer)));
}

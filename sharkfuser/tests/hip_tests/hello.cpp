#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <hip/hip_runtime.h>

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

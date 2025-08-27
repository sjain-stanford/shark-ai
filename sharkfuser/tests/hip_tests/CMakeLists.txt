enable_language(HIP)  # enables HIP language for this dir and this dir only

include(Catch)

set_source_files_properties(test_hip_buffer_import_api.cpp PROPERTIES LANGUAGE HIP)

add_sharkfuser_test(
  NAME sharkfuser_hip_tests
  SRCS
    test_hip_buffer_import_api.cpp
  DEPS
    iree_runtime_unified
    Catch2::Catch2WithMain
)

set_target_properties(sharkfuser_hip_tests PROPERTIES CMAKE_HIP_ARCHITECTURES "gfx942")
target_compile_features(sharkfuser_hip_tests PRIVATE cxx_std_20)

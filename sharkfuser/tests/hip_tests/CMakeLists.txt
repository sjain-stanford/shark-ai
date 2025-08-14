enable_language(HIP)  # enables HIP language for this dir and this dir only

include(Catch)

set_source_files_properties(hello.cpp PROPERTIES LANGUAGE HIP)

add_sharkfuser_test(
  NAME hello
  SRCS
    hello.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

set_target_properties(hello PROPERTIES CMAKE_HIP_ARCHITECTURES "gfx942")
target_compile_features(hello PRIVATE cxx_std_20)

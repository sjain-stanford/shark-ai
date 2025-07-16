// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_EMIT_ASM_H
#define FUSILI_EMIT_ASM_H

#include <format>

#include "fusili/graph.h"
#include "fusili/node/conv_node.h"

namespace fusili {

// We use a combination of raw multi-line strings `R"(...)"` and `std::format`
// (from c++20) to implement a simple templating system for generating mlir
// assembly code. This could be made better with a more sophisticated
// jinja2-like templating system at some point. For now this gets us
// mostly what we need.

// Caution: An important foot-gun here is to forget to double the brace for
// a literal `{` or `}`. i.e. always use `{{` for `{` and `}}` for `}` to
// disambiguate from the `{}` that `std::format` uses for replacements.
// If not you'll hit a compilation error like so:
//    "error: call to consteval function 'std::basic_format_string<char, ...'"
//    "is not a constant expression"

inline std::string Graph::emit_asm_node_pre() {
  constexpr std::string_view str = R"(
module @module {{
  func.func @{}(%arg0: !torch.vtensor<[16,128,64,64],f32>, %arg1: !torch.vtensor<[256,128,1,1],f32>) -> !torch.vtensor<[16,256,64,64],f32> attributes {{torch.assume_strict_symbolic_shapes}} {{
  )";

  std::string output = std::format(str, "main");
  return output;
}

inline std::string Graph::emit_asm_node_post() {
  return R"(
    return %4 : !torch.vtensor<[16,256,64,64],f32>
  }
}
  )";
}

inline std::string ConvFPropNode::emit_asm_node_pre() {
  return R"(
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %4 = torch.aten.convolution %arg0, %arg1, %none, %0, %1, %2, %false, %3, %int1 : !torch.vtensor<[16,128,64,64],f32>, !torch.vtensor<[256,128,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[16,256,64,64],f32>
    )";
}

inline std::string ConvFPropNode::emit_asm_node_post() { return ""; }

} // namespace fusili

#endif // FUSILI_EMIT_ASM_H

// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_NODE_NODE_H
#define FUSILI_NODE_NODE_H

#include <memory>
#include <sstream>

#include "fusili/context.h"
#include "fusili/logging.h"

namespace fusili {

class INode {
private:
  virtual error_t pre_validate_node() const { return {error_code_t::OK, ""}; }

  virtual error_t infer_properties_node() = 0;

  virtual error_t post_validate_node() const { return {error_code_t::OK, ""}; }

  virtual std::string emit_asm_node_pre() = 0;
  virtual std::string emit_asm_node_post() = 0;

protected:
  enum class Type {
    GRAPH,
    COMPOSITE,
    CONVOLUTION,
  };
  Type tag;

  // This is a list of sub-nodes that this node may contain.
  // This is implicitly topologically sorted, as a result of
  // the functional API.
  std::vector<std::shared_ptr<INode>> sub_nodes;

  error_t validate_subtree() {
    FUSILI_CHECK_ERROR(pre_validate_node());
    FUSILI_CHECK_ERROR(infer_properties_node());
    for (const auto &sub_node : sub_nodes) {
      FUSILI_CHECK_ERROR(sub_node->validate_subtree());
    }
    FUSILI_CHECK_ERROR(post_validate_node());
    return {error_code_t::OK, ""};
  }

  void emit_asm_subtree(std::ostringstream &oss) {
    oss << emit_asm_node_pre();
    for (const auto &sub_node : sub_nodes) {
      sub_node->emit_asm_subtree(oss);
    }
    oss << emit_asm_node_post();
  }

public:
  Context context;

  INode(Context const &ctx) : context(ctx) {}
  virtual ~INode() = default;

  virtual Type getType() = 0;
};

template <typename DerivedT> class NodeCRTP : public INode {
private:
  DerivedT &self() { return static_cast<DerivedT &>(*this); }
  const DerivedT &self() const { return static_cast<const DerivedT &>(*this); }

protected:
  // Allow derived NodeCRTP classes to use the INode constructor
  using INode::INode;
};

} // namespace fusili

#endif // FUSILI_NODE_NODE_H

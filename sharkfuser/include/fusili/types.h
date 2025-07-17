// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_TYPES_H
#define FUSILI_TYPES_H

#include <string>
#include <unordered_map>

namespace fusili {

enum class DataType_t {
  NOT_SET,
  HALF,
  BFLOAT16,
  FLOAT,
  DOUBLE,
  UINT8,
  INT8,
  INT16,
  INT32,
  INT64,
  BOOLEAN,
  FP8_E5M2,
};

static const std::unordered_map<DataType_t, std::string>
    DATA_TYPE_TO_MLIR_TYPE = {
        {DataType_t::HALF, "f16"},        {DataType_t::BFLOAT16, "bf16"},
        {DataType_t::FLOAT, "f32"},       {DataType_t::DOUBLE, "f64"},
        {DataType_t::UINT8, "ui8"},       {DataType_t::INT8, "si8"},
        {DataType_t::INT16, "si16"},      {DataType_t::INT32, "si32"},
        {DataType_t::INT64, "si64"},      {DataType_t::BOOLEAN, "i1"},
        {DataType_t::FP8_E5M2, "f8E5M2"},
};

} // namespace fusili

#endif // FUSILI_TYPES_H

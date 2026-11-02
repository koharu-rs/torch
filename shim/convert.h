// Conversions between koharu-torch's C types and ATen's that libtorch's
// AOTInductor helpers in torch/csrc/inductor/aoti_torch/utils.h lack.

#pragma once

#include "koharu_torch.h"

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <vector>

// An array the caller frees with aten_free.
template <class T>
T* aten_new_array(size_t len) {
  auto* array = static_cast<T*>(std::malloc(std::max<size_t>(len, 1) * sizeof(T)));
  TORCH_CHECK(array, "out of host memory for ", len, " elements");
  return array;
}

inline c10::Scalar aten_scalar(AtenScalar scalar) {
  switch (scalar.type) {
    case ATEN_SCALAR_BOOL:
      return scalar.value.b;
    case ATEN_SCALAR_INT:
      return scalar.value.i;
    case ATEN_SCALAR_FLOAT:
      return scalar.value.f;
    case ATEN_SCALAR_COMPLEX:
      return c10::complex<double>(scalar.value.c.real, scalar.value.c.imag);
  }
  TORCH_CHECK(false, "a scalar of unknown type ", static_cast<int>(scalar.type));
}

inline std::optional<c10::Scalar> aten_scalar(
    const AtenScalar* scalar) {
  return scalar ? std::make_optional(aten_scalar(*scalar))
                : std::nullopt;
}

inline std::vector<c10::Scalar> aten_scalars(
    const AtenScalar* scalars,
    int64_t len) {
  std::vector<c10::Scalar> result;
  result.reserve(len);
  for (int64_t i = 0; i < len; ++i) {
    result.push_back(aten_scalar(scalars[i]));
  }
  return result;
}

inline AtenScalar aten_from_scalar(const c10::Scalar& scalar) {
  AtenScalar result{};
  if (scalar.isBoolean()) {
    result.type = ATEN_SCALAR_BOOL;
    result.value.b = scalar.toBool();
  } else if (scalar.isIntegral(/*includeBool=*/false)) {
    result.type = ATEN_SCALAR_INT;
    result.value.i = scalar.toLong();
  } else if (scalar.isComplex()) {
    auto value = scalar.toComplexDouble();
    result.type = ATEN_SCALAR_COMPLEX;
    result.value.c.real = value.real();
    result.value.c.imag = value.imag();
  } else {
    result.type = ATEN_SCALAR_FLOAT;
    result.value.f = scalar.toDouble();
  }
  return result;
}

inline void aten_tensor_list(
    std::vector<at::Tensor>&& tensors,
    AtenTensorHandle** ret,
    int64_t* len) {
  *ret = aten_new_array<AtenTensorHandle>(tensors.size());
  *len = static_cast<int64_t>(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    (*ret)[i] = torch::aot_inductor::new_tensor_handle(std::move(tensors[i]));
  }
}

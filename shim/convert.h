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

// An array the caller frees with koharu_torch_free.
template <class T>
T* koharu_torch_new_array(size_t len) {
  auto* array = static_cast<T*>(std::malloc(std::max<size_t>(len, 1) * sizeof(T)));
  TORCH_CHECK(array, "out of host memory for ", len, " elements");
  return array;
}

inline c10::Scalar koharu_torch_scalar(KoharuTorchScalar scalar) {
  switch (scalar.type) {
    case KOHARU_TORCH_SCALAR_BOOL:
      return scalar.value.b;
    case KOHARU_TORCH_SCALAR_INT:
      return scalar.value.i;
    case KOHARU_TORCH_SCALAR_FLOAT:
      return scalar.value.f;
    case KOHARU_TORCH_SCALAR_COMPLEX:
      return c10::complex<double>(scalar.value.c.real, scalar.value.c.imag);
  }
  TORCH_CHECK(false, "a scalar of unknown type ", static_cast<int>(scalar.type));
}

inline std::optional<c10::Scalar> koharu_torch_scalar(
    const KoharuTorchScalar* scalar) {
  return scalar ? std::make_optional(koharu_torch_scalar(*scalar))
                : std::nullopt;
}

inline std::vector<c10::Scalar> koharu_torch_scalars(
    const KoharuTorchScalar* scalars,
    int64_t len) {
  std::vector<c10::Scalar> result;
  result.reserve(len);
  for (int64_t i = 0; i < len; ++i) {
    result.push_back(koharu_torch_scalar(scalars[i]));
  }
  return result;
}

inline KoharuTorchScalar koharu_torch_from_scalar(const c10::Scalar& scalar) {
  KoharuTorchScalar result{};
  if (scalar.isBoolean()) {
    result.type = KOHARU_TORCH_SCALAR_BOOL;
    result.value.b = scalar.toBool();
  } else if (scalar.isIntegral(/*includeBool=*/false)) {
    result.type = KOHARU_TORCH_SCALAR_INT;
    result.value.i = scalar.toLong();
  } else if (scalar.isComplex()) {
    auto value = scalar.toComplexDouble();
    result.type = KOHARU_TORCH_SCALAR_COMPLEX;
    result.value.c.real = value.real();
    result.value.c.imag = value.imag();
  } else {
    result.type = KOHARU_TORCH_SCALAR_FLOAT;
    result.value.f = scalar.toDouble();
  }
  return result;
}

inline void koharu_torch_tensor_list(
    std::vector<at::Tensor>&& tensors,
    AtenTensorHandle** ret,
    int64_t* len) {
  *ret = koharu_torch_new_array<AtenTensorHandle>(tensors.size());
  *len = static_cast<int64_t>(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    (*ret)[i] = torch::aot_inductor::new_tensor_handle(std::move(tensors[i]));
  }
}

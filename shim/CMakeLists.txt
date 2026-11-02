cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(koharu-torch CXX)

set(PYTORCH_SOURCE "" CACHE PATH "The PyTorch source tree libtorch was built from")
if(NOT EXISTS "${PYTORCH_SOURCE}/aten/src/ATen/native/native_functions.yaml")
  message(FATAL_ERROR "PYTORCH_SOURCE must name the PyTorch source tree")
endif()
set(LIBTORCH "" CACHE PATH "The libtorch tree to build against, with include and lib")

# The shim calls only torch_cpu and c10, so it links them alone: libtorch's
# own CMake package would also need the CUDA or ROCm toolkit a GPU build
# links.
find_library(TORCH_CPU torch_cpu PATHS "${LIBTORCH}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(C10 c10 PATHS "${LIBTORCH}/lib" NO_DEFAULT_PATH REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(generated
  "${CMAKE_CURRENT_BINARY_DIR}/koharu_torch_aten.h"
  "${CMAKE_CURRENT_BINARY_DIR}/koharu_torch_aten.cpp")
add_custom_command(
  OUTPUT ${generated}
  COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"
    "${PYTORCH_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS generate.py
    "${PYTORCH_SOURCE}/aten/src/ATen/native/native_functions.yaml"
    "${PYTORCH_SOURCE}/aten/src/ATen/native/tags.yaml")

add_library(koharu-torch SHARED koharu_torch.cpp ${generated})
target_include_directories(koharu-torch PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
  "${LIBTORCH}/include" "${LIBTORCH}/include/torch/csrc/api/include")
target_compile_definitions(koharu-torch PRIVATE KOHARU_TORCH_BUILD)
if(MSVC)
  # What libtorch's CMake package gives an MSVC consumer.
  target_compile_options(koharu-torch PRIVATE /permissive- /EHsc /bigobj)
endif()
target_link_libraries(koharu-torch PRIVATE "${TORCH_CPU}" "${C10}")
set_property(TARGET koharu-torch PROPERTY CXX_STANDARD 20)
install(TARGETS koharu-torch DESTINATION .)
install(FILES koharu_torch.h "${CMAKE_CURRENT_BINARY_DIR}/koharu_torch_aten.h"
  DESTINATION include)

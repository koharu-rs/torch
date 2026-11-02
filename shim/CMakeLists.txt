cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(koharu-torch CXX)

set(PYTORCH_SOURCE "" CACHE PATH "The PyTorch source tree libtorch was built from")
if(NOT EXISTS "${PYTORCH_SOURCE}/aten/src/ATen/native/native_functions.yaml")
  message(FATAL_ERROR "PYTORCH_SOURCE must name the PyTorch source tree")
endif()

find_package(Torch REQUIRED)
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
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(koharu-torch PRIVATE KOHARU_TORCH_BUILD)
target_link_libraries(koharu-torch "${TORCH_LIBRARIES}")
set_property(TARGET koharu-torch PROPERTY CXX_STANDARD 20)
install(TARGETS koharu-torch DESTINATION .)
install(FILES koharu_torch.h "${CMAKE_CURRENT_BINARY_DIR}/koharu_torch_aten.h"
  DESTINATION include)

cmake_minimum_required(VERSION 3.20)
project(vocalarc_onnx_sidecar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ORT_ROOT "" CACHE PATH "ONNX Runtime SDK root")

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ORT_ROOT}/include" "${ORT_ROOT}/include/onnxruntime"
  REQUIRED)
find_library(ORT_LIBRARY NAMES onnxruntime onnxruntime.lib
  HINTS "${ORT_ROOT}/lib" "${ORT_ROOT}/lib64"
  REQUIRED)

add_executable(vocalarc-onnx-sidecar onnx_sidecar.cpp)
target_include_directories(vocalarc-onnx-sidecar PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(vocalarc-onnx-sidecar PRIVATE "${ORT_LIBRARY}")
if(MSVC)
  target_compile_options(vocalarc-onnx-sidecar PRIVATE /O2 /EHsc /W4)
else()
  target_compile_options(vocalarc-onnx-sidecar PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

if(UNIX AND NOT APPLE)
  set_target_properties(vocalarc-onnx-sidecar PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
  set_target_properties(vocalarc-onnx-sidecar PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
endif()

include(GNUInstallDirs)
install(TARGETS vocalarc-onnx-sidecar RUNTIME DESTINATION runtime LIBRARY DESTINATION runtime)

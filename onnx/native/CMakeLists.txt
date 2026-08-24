cmake_minimum_required(VERSION 3.20)
project(vocalarc_onnx_sidecar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ORT_ROOT "" CACHE PATH "ONNX Runtime SDK root")
option(VOCALARC_ENABLE_CUDA_DSP "Build the GPU-resident CUDA STFT/ISTFT path" OFF)

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ORT_ROOT}/include" "${ORT_ROOT}/include/onnxruntime"
  REQUIRED)
find_library(ORT_LIBRARY NAMES onnxruntime onnxruntime.lib
  HINTS "${ORT_ROOT}/lib" "${ORT_ROOT}/lib64"
  REQUIRED)

add_executable(vocalarc-onnx-sidecar onnx_sidecar.cpp)
target_include_directories(vocalarc-onnx-sidecar PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(vocalarc-onnx-sidecar PRIVATE "${ORT_LIBRARY}")
if(VOCALARC_ENABLE_CUDA_DSP)
  if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(VOCALARC_REQUESTED_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(vocalarc-onnx-sidecar PRIVATE cuda_dsp.cu)
  target_link_libraries(vocalarc-onnx-sidecar PRIVATE CUDA::cudart CUDA::cufft)
  target_compile_definitions(vocalarc-onnx-sidecar PRIVATE VOCALARC_ENABLE_CUDA_DSP=1)
  set_target_properties(vocalarc-onnx-sidecar PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(vocalarc-onnx-sidecar PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  if(VOCALARC_REQUESTED_CUDA_ARCHITECTURES)
    set_property(TARGET vocalarc-onnx-sidecar PROPERTY CUDA_ARCHITECTURES
      "${VOCALARC_REQUESTED_CUDA_ARCHITECTURES}")
  else()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      set_property(TARGET vocalarc-onnx-sidecar PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86 89 90 100 120)
    else()
      set_property(TARGET vocalarc-onnx-sidecar PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90 100 120)
    endif()
  endif()
endif()
if(MSVC)
  target_compile_options(vocalarc-onnx-sidecar PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(vocalarc-onnx-sidecar PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()

if(UNIX AND NOT APPLE)
  set_target_properties(vocalarc-onnx-sidecar PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
  set_target_properties(vocalarc-onnx-sidecar PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
endif()

include(GNUInstallDirs)
install(TARGETS vocalarc-onnx-sidecar RUNTIME DESTINATION runtime LIBRARY DESTINATION runtime)

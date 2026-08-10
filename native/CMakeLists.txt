cmake_minimum_required(VERSION 3.16)
project(vocalarc_separation C)

option(VOCALARC_NATIVE_FAST "Enable compiler host tuning for a local benchmark" OFF)
option(VOCALARC_CUDA "Build the optional fused CUDA provider" OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(VOCALARC_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/assets/model.f32")
if(NOT EXISTS "${VOCALARC_MODEL}")
  message(FATAL_ERROR "Missing bundled model: ${VOCALARC_MODEL}")
endif()

# The raw executable is never shipped. The embed targets append the model and
# preserve the executable's loader image, producing one self-contained file.
add_executable(vocalarc-embed-model EXCLUDE_FROM_ALL embed_model.c)
if(MSVC)
  target_compile_options(vocalarc-embed-model PRIVATE /O2)
  set_property(TARGET vocalarc-embed-model PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
else()
  target_compile_options(vocalarc-embed-model PRIVATE -O2)
endif()

function(configure_cpu_target target)
  if(NOT MSVC)
    target_compile_definitions(${target} PRIVATE _POSIX_C_SOURCE=200809L)
    target_compile_options(${target} PRIVATE -O3 -ffast-math -fno-math-errno -funroll-loops)
    target_link_libraries(${target} PRIVATE m)
  else()
    target_compile_options(${target} PRIVATE /O2 /fp:fast)
    set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  if(VOCALARC_NATIVE_FAST AND NOT MSVC)
    target_compile_options(${target} PRIVATE -march=native)
  endif()
  if(WIN32)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_link_libraries(${target} PRIVATE pthread ${CMAKE_DL_LIBS})
  endif()
endfunction()

add_executable(vocalarc-separation-core vocalarc_separation.c native_gpu.c)
configure_cpu_target(vocalarc-separation-core)

set(VOCALARC_CPU_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vocalarc-separation${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_command(
  OUTPUT "${VOCALARC_CPU_OUTPUT}"
  COMMAND $<TARGET_FILE:vocalarc-embed-model>
          $<TARGET_FILE:vocalarc-separation-core>
          "${VOCALARC_MODEL}"
          "${VOCALARC_CPU_OUTPUT}"
  DEPENDS vocalarc-embed-model vocalarc-separation-core "${VOCALARC_MODEL}"
  VERBATIM
)
add_custom_target(vocalarc-separation ALL DEPENDS "${VOCALARC_CPU_OUTPUT}")

if(VOCALARC_CUDA)
  enable_language(CUDA)
  add_executable(vocalarc-separation-gpu-core vocalarc_separation.c native_gpu_cuda.cu)
  set_target_properties(vocalarc-separation-gpu-core PROPERTIES
    CUDA_STANDARD 14
    CUDA_SEPARABLE_COMPILATION ON
  )
  configure_cpu_target(vocalarc-separation-gpu-core)
  target_link_libraries(vocalarc-separation-gpu-core PRIVATE cublas)

  set(VOCALARC_GPU_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vocalarc-separation-gpu${CMAKE_EXECUTABLE_SUFFIX}")
  add_custom_command(
    OUTPUT "${VOCALARC_GPU_OUTPUT}"
    COMMAND $<TARGET_FILE:vocalarc-embed-model>
            $<TARGET_FILE:vocalarc-separation-gpu-core>
            "${VOCALARC_MODEL}"
            "${VOCALARC_GPU_OUTPUT}"
    DEPENDS vocalarc-embed-model vocalarc-separation-gpu-core "${VOCALARC_MODEL}"
    VERBATIM
  )
  add_custom_target(vocalarc-separation-gpu ALL DEPENDS "${VOCALARC_GPU_OUTPUT}")
endif()

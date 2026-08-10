ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD := $(ROOT)/build
CC ?= cc
CFLAGS ?= -O3 -DNDEBUG -std=c11 -D_POSIX_C_SOURCE=200809L -ffast-math -fno-math-errno -fno-trapping-math -funroll-loops -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion
PYTHON ?= python3
NVCC ?= nvcc
CUDA_ARCH ?= compute_75
LDFLAGS ?=
LDLIBS ?=

UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),Darwin)
  LDLIBS += -lm
else ifeq ($(UNAME_S),Linux)
  LDLIBS += -lm -pthread -ldl
endif

.PHONY: all gpu model clean test benchmark

all: $(BUILD)/vocalarc-separation

gpu: $(BUILD)/vocalarc-separation-gpu

$(BUILD):
	mkdir -p $@

$(BUILD)/embed-model: $(ROOT)/embed_model.c | $(BUILD)
	$(CC) $(CFLAGS) -O2 -o $@ $(ROOT)/embed_model.c

$(BUILD)/vocalarc-separation-core: $(ROOT)/vocalarc_separation.c $(ROOT)/native_gpu.c $(ROOT)/native_gpu.h $(ROOT)/native_cpu_features.c $(ROOT)/native_simd.c $(ROOT)/native_simd.h | $(BUILD)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(ROOT)/vocalarc_separation.c $(ROOT)/native_gpu.c $(ROOT)/native_cpu_features.c $(ROOT)/native_simd.c $(LDLIBS)

$(BUILD)/vocalarc-separation: $(BUILD)/vocalarc-separation-core $(BUILD)/embed-model $(ROOT)/assets/model.f32
	$(BUILD)/embed-model $(BUILD)/vocalarc-separation-core $(ROOT)/assets/model.f32 $@

$(BUILD)/vocalarc-gpu-host.o: $(ROOT)/vocalarc_separation.c $(ROOT)/native_gpu.h $(ROOT)/native_simd.h | $(BUILD)
	$(CC) $(CFLAGS) -c $(ROOT)/vocalarc_separation.c -o $@

$(BUILD)/vocalarc-native-simd.o: $(ROOT)/native_simd.c $(ROOT)/native_simd.h | $(BUILD)
	$(CC) $(CFLAGS) -c $(ROOT)/native_simd.c -o $@

$(BUILD)/vocalarc-cpu-features.o: $(ROOT)/native_cpu_features.c $(ROOT)/native_simd.h | $(BUILD)
	$(CC) $(CFLAGS) -c $(ROOT)/native_cpu_features.c -o $@

$(BUILD)/native_gpu_cuda.o: $(ROOT)/native_gpu_cuda.cu $(ROOT)/native_gpu.h | $(BUILD)
	$(NVCC) -O3 --use_fast_math -std=c++14 -arch=$(CUDA_ARCH) -c $(ROOT)/native_gpu_cuda.cu -o $@

$(BUILD)/vocalarc-separation-gpu-core: $(BUILD)/vocalarc-gpu-host.o $(BUILD)/native_gpu_cuda.o $(BUILD)/vocalarc-cpu-features.o $(BUILD)/vocalarc-native-simd.o | $(BUILD)
	$(NVCC) -O3 --use_fast_math -arch=$(CUDA_ARCH) $^ -lcublas -o $@

$(BUILD)/vocalarc-separation-gpu: $(BUILD)/vocalarc-separation-gpu-core $(BUILD)/embed-model $(ROOT)/assets/model.f32
	$(BUILD)/embed-model $(BUILD)/vocalarc-separation-gpu-core $(ROOT)/assets/model.f32 $@

model:
	@test -f $(ROOT)/assets/model.f32 || (echo "missing bundled assets/model.f32" >&2; exit 1)

test: all model
	sh $(ROOT)/scripts/test_native.sh $(BUILD)/vocalarc-separation $(ROOT)/assets/model.f32
	$(PYTHON) $(ROOT)/scripts/benchmark.py --native $(BUILD)/vocalarc-separation --model $(ROOT)/assets/model.f32 --seconds 0.1 --threads 1 --warmups 0 --repetitions 1

benchmark: all model
	$(PYTHON) $(ROOT)/scripts/benchmark.py --native $(BUILD)/vocalarc-separation --model $(ROOT)/assets/model.f32

clean:
	rm -rf $(BUILD)

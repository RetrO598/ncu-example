#pragma once
// #include <__clang_cuda_runtime_wrapper.h>
#include <cstddef>
#include <curand_mtgp32_kernel.h>

constexpr int arraySize = 1024;

__global__ void reductionNaive(float *input, float *output, size_t n) {
  size_t i = threadIdx.x;
  for (size_t s = 1; s < n; s *= 2) {
    if (i % s == 0) {
      input[i] += input[i + s];
    }
    __syncthreads();
  }
  if (i == 0) {
    *output = input[i];
  }
}

__global__ void reductionSerial(float *input, float *output, size_t n) {
  size_t i = threadIdx.x;
  for (size_t stride = blockDim.x; stride >= 1; stride /= 2) {
    if (i < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  if (i == 0) {
    *output = input[0];
  }
}

template <size_t BLOCKSIZE>
__global__ void reductionShared(float *input, float *output, size_t n) {
  size_t i = threadIdx.x;
  __shared__ float input_s[BLOCKSIZE];
  input_s[i] = input[i] + input[i + BLOCKSIZE];

  for (size_t stride = BLOCKSIZE / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (i < stride) {
      input_s[i] = input_s[i] + input_s[i + stride];
    }
  }

  if (i == 0) {
    *output = input_s[0];
  }
}
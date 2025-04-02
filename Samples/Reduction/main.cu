#include "kernels.cuh"
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <helper_functions.h>
#include <iostream>
#include <vector_types.h>

float cpuReduction(float *input, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum += input[i];
  }
  return sum;
}

int main() {
  size_t memSize = arraySize * sizeof(float);
  float *input_h = new float[memSize];
  float *input_d;
  float *output_d;

  checkCudaErrors(cudaMalloc((void **)&input_d, memSize));
  checkCudaErrors(cudaMalloc((void **)&output_d, sizeof(float)));

  std::srand(std::time(0));

  for (size_t i = 0; i < arraySize; ++i) {
    input_h[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }

  checkCudaErrors(
      cudaMemcpy(input_d, input_h, memSize, cudaMemcpyHostToDevice));

  dim3 grid(1, 1, 1);
  dim3 block(arraySize / 2, 1, 1);

  reductionShared<arraySize / 2><<<grid, block>>>(input_d, output_d, arraySize);

  checkCudaErrors(cudaGetLastError());
  float *gpu_result = new float;
  checkCudaErrors(
      cudaMemcpy(gpu_result, output_d, sizeof(float), cudaMemcpyDeviceToHost));
  float output_h = cpuReduction(input_h, arraySize);
  std::cout << "CPU Sum: " << *gpu_result << "\n";
  std::cout << "GPU Sum: " << output_h << "\n";
  if (fabs(*gpu_result - output_h) < 1e-4) {
    std::cout << "Results match!" << std::endl;
  } else {
    std::cout << "Mismatch detected!" << std::endl;
  }
}
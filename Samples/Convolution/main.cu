#include "kernels.cuh"
#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <helper_functions.h>
#include <iostream>
#include <vector_types.h>

constexpr int width = 1024;
constexpr int height = 1000;
constexpr int NSTEPS = 100;

__host__ void convolution(float *input, float *filter, float *output) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float value = 0.0f;
      for (int j = 0; j < 2 * filterRadius + 1; ++j) {
        for (int i = 0; i < 2 * filterRadius + 1; ++i) {
          int inputx = x - filterRadius + i;
          int inputy = y - filterRadius + j;
          if (inputx >= 0 && inputx < width && inputy >= 0 && inputy < height) {
            value +=
                filter[j * filterRadius + i] * input[inputy * width + inputx];
          }
        }
      }
      output[y * width + x] = value;
    }
  }
}

bool checkAnswer(float *output_d, float *output_h) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      if (output_d[j * width + i] != output_h[j * width + i]) {
        return false;
      }
    }
  }
  return true;
}

void printOut(float *output) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      std::cout << output[j * width + i] << " ";
    }
    std::cout << "\n";
  }
}
void printFilter(float *filter) {
  for (int j = 0; j < 2 * filterRadius + 1; ++j) {
    for (int i = 0; i < 2 * filterRadius + 1; ++i) {
      std::cout << filter[j * (2 * filterRadius + 1) + i] << " ";
    }
    std::cout << "\n";
  }
}

int main() {
  size_t memsize = sizeof(float) * width * height;
  size_t filterSize =
      sizeof(float) * (2 * filterRadius + 1) * (2 * filterRadius + 1);
  float *input_h = new float[width * height];
  float *output_h = new float[width * height];
  float *filter_h = new float[(2 * filterRadius + 1) * (2 * filterRadius + 1)];

  float *input_d;
  float *output_d;
  float *filter_d;

  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      input_h[j * width + i] = i + j;
    }
  }

  for (int j = 0; j < 2 * filterRadius + 1; ++j) {
    for (int i = 0; i < 2 * filterRadius + 1; ++i) {
      filter_h[j * (2 * filterRadius + 1) + i] = 1;
    }
  }

  convolution(input_h, filter_h, output_h);

  checkCudaErrors(cudaMalloc((void **)&input_d, memsize));
  checkCudaErrors(cudaMalloc((void **)&output_d, memsize));
  checkCudaErrors(cudaMalloc((void **)&filter_d, filterSize));

  checkCudaErrors(
      cudaMemcpy(input_d, input_h, memsize, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(filter_d, filter_h, filterSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(constFilter, filter_h, filterSize));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  dim3 Grid((width + OUTPUT_TILE - 1) / OUTPUT_TILE,
            (height + OUTPUT_TILE - 1) / OUTPUT_TILE, 1);
  dim3 Block(BLOCKSIZE, BLOCKSIZE, 1);
  convolutionShared<<<Grid, Block>>>(input_d, output_d, width, height);
  checkCudaErrors(cudaGetLastError());
  convolutionShared<<<Grid, Block>>>(input_d, output_d, width, height);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaEventRecord(start));
  for (int t = 0; t < NSTEPS; ++t) {
    convolutionShared<<<Grid, Block>>>(input_d, output_d, width, height);
    checkCudaErrors(cudaGetLastError());
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time;
  checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
  std::cout << "time：" << time / NSTEPS << "ms\n";

  float *result = new float[memsize];
  checkCudaErrors(
      cudaMemcpy(result, output_d, memsize, cudaMemcpyDeviceToHost));

  if (checkAnswer(result, output_h)) {
    std::cout << "right answer" << "\n";
  }
}
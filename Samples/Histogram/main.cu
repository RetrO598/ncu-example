#include "helper_functions.h"
#include "kernels.cuh"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <vector_types.h>

constexpr unsigned int STRING_LEGTH = 1 << 26;

constexpr int NTIMES = 100;

void generateRandomString(char *str, unsigned int length) {
  for (unsigned int i = 0; i < length; i++) {
    str[i] = 'a' + (rand() % 26);
  }
}

void histogramCPU(char *data, unsigned int length, unsigned int *histo) {
  for (unsigned int i = 0; i < length; i++) {
    int position = data[i] - 'a';
    if (position >= 0 && position < 26) {
      histo[position]++;
    }
  }
}

bool compareResults(unsigned int *cpu, unsigned int *gpu, unsigned int length) {
  for (int i = 0; i < length; ++i) {
    if (cpu[i] != gpu[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  srand(time(NULL));

  char *h_data = (char *)malloc(STRING_LEGTH * sizeof(char));

  unsigned int h_histo[HISTO_BIN];
  unsigned int h_histo_gpu[HISTO_BIN];

  for (int i = 0; i < HISTO_BIN; ++i) {
    h_histo[i] = 0;
  }

  generateRandomString(h_data, STRING_LEGTH);

  histogramCPU(h_data, STRING_LEGTH, h_histo);

  char *d_data;
  unsigned int *d_histo;

  checkCudaErrors(cudaMalloc(&d_data, STRING_LEGTH * sizeof(char)));

  checkCudaErrors(cudaMalloc(&d_histo, HISTO_BIN * sizeof(unsigned int)));

  checkCudaErrors(cudaMemcpy(d_data, h_data, STRING_LEGTH * sizeof(char),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemset(d_histo, 0, HISTO_BIN * sizeof(unsigned int)));

  dim3 block;
  dim3 grid;
  auto kernel = &histogramNaive;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  for (int i = 0; i < 5; ++i) {
    if (i == 0) {
      printf("Testing Kernel histogramNaive......\n");
      grid = dim3((STRING_LEGTH + BLOCK_SIZE - 1) / BLOCK_SIZE);
      block = dim3(BLOCK_SIZE, 1, 1);
    } else if (i == 1) {
      kernel = &histogramPrivatization;
      printf("Testing Kernel histogramPrivatization......\n");
      grid = dim3((STRING_LEGTH + BLOCK_SIZE - 1) / BLOCK_SIZE);
      block = dim3(BLOCK_SIZE, 1, 1);
    } else if (i == 2) {
      constexpr int coarsen = 4;
      grid = dim3((STRING_LEGTH + BLOCK_SIZE * coarsen - 1) /
                  (BLOCK_SIZE * coarsen));
      block = dim3(BLOCK_SIZE, 1, 1);
      printf("Testing Kernel histogramCoarsenContiguous......\n");
      kernel = &histogramCoarsenContiguous<coarsen>;
    } else if (i == 3) {
      constexpr int coarsen = 4;
      grid = dim3((STRING_LEGTH + BLOCK_SIZE * coarsen - 1) /
                  (BLOCK_SIZE * coarsen));
      block = dim3(BLOCK_SIZE, 1, 1);
      printf("Testing Kernel histogramCoarsenInterleaved......\n");
      kernel = &histogramCoarsenInterleaved;
    } else {
      constexpr int coarsen = 4;
      grid = dim3((STRING_LEGTH + BLOCK_SIZE * coarsen - 1) /
                  (BLOCK_SIZE * coarsen));
      block = dim3(BLOCK_SIZE, 1, 1);
      printf("Testing Kernel histogramAggregation......\n");
      kernel = &histogramAggregation;
    }

    // warm up
    kernel<<<grid, block>>>(d_data, STRING_LEGTH, d_histo);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(start, 0));
    for (int i = 0; i < NTIMES; ++i) {
      kernel<<<grid, block>>>(d_data, STRING_LEGTH, d_histo);
      checkCudaErrors(cudaGetLastError());
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float time;
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    printf("Elapsed time: %f ms\n", time / NTIMES);

    checkCudaErrors(cudaMemcpy(h_histo_gpu, d_histo,
                               HISTO_BIN * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));

    if (compareResults(h_histo, h_histo_gpu, HISTO_BIN)) {
      printf("right answer\n");
    }
  }

  free(h_data);
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_histo));

  return 0;
}
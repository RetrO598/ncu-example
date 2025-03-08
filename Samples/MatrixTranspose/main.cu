#include "kernels.cuh"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <helper_functions.h>

#include <iostream>

// Current Problems：
// 1. No completely free of bank conflicts

bool checkAnswer(float *matA, float *matB, const int &NX, const int &NY) {
  for (int j = 0; j < NY; ++j) {
    for (int i = 0; i < NX; ++i) {
      if (matA[i * NY + j] != matB[j * NX + i]) {
        return false;
      }
    }
  }
  return true;
}

void printMat(float *mat, const int &NX, const int &NY) {
  for (int j = 0; j < NY; ++j) {
    for (int i = 0; i < NX; ++i) {
      std::cout << mat[j * NX + i] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main() {
  const int NX = 1024;
  const int NY = 1024;
  const int NTIMES = 1;

  const int BLOCKSIZE = 32;
  dim3 Grid((NX + BLOCKSIZE - 1) / BLOCKSIZE, (NY + BLOCKSIZE - 1) / BLOCKSIZE,
            1);
  dim3 Block(BLOCKSIZE, BLOCKSIZE, 1);

  float *matA_d;
  float *matB_d;
  checkCudaErrors(cudaMalloc((void **)&matA_d, sizeof(float) * NX * NY));
  checkCudaErrors(cudaMalloc((void **)&matB_d, sizeof(float) * NX * NY));

  float *matA_h = new float[NX * NY];
  float *matB_h = new float[NX * NY];
  float *ans_h = new float[NX * NY];

  for (int j = 0; j < NY; ++j) {
    for (int i = 0; i < NX; ++i) {
      matB_h[j * NX + i] = j * NX + i;
    }
  }

  checkCudaErrors(cudaMemcpy(matB_d, matB_h, sizeof(float) * NX * NY,
                             cudaMemcpyHostToDevice));

  // transpose on host
  for (int j = 0; j < NY; ++j) {
    for (int i = 0; i < NX; ++i) {
      matA_h[i * NY + j] = matB_h[j * NX + i];
    }
  }

  auto kernel = &matrixCopyShared<BLOCKSIZE>;
  auto mem_size = static_cast<size_t>(sizeof(float) * NX * NY);
  cudaEvent_t start, end;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));

  for (int i = 0; i < 5; ++i) {
    if (i == 0) {
      printf("Testing Kernel matrixCopyShared......\n");
    } else if (i == 1) {
      printf("Testing Kernel matrixTransposeNaive......\n");
      kernel = &matrixTransposeNaive;
    } else if (i == 2) {
      printf("Testing Kernel matrixTransposeShared......\n");
      kernel = &matrixTransposeShared<BLOCKSIZE>;
    } else if (i == 3) {
      printf("Testing Kernel matrixTransposeTransposeSharedPadding......\n");
      kernel = &matrixTransposeSharedPadding<BLOCKSIZE>;
    } else {
      printf("Testing Kernel matrixTransposeSharedSwizz......\n");
      kernel = &matrixTransposeSharedSwizz<BLOCKSIZE>;
    }
    // warm up
    kernel<<<Grid, Block>>>(matA_d, matB_d, NX, NY);
    checkCudaErrors(cudaGetLastError());

    // start time measurements
    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < NTIMES; ++i) {
      kernel<<<Grid, Block>>>(matA_d, matB_d, NX, NY);
      checkCudaErrors(cudaGetLastError());
    }

    checkCudaErrors(cudaEventRecord(end));
    checkCudaErrors(cudaEventSynchronize(end));

    float kernelTime;
    checkCudaErrors(cudaEventElapsedTime(&kernelTime, start, end));

    float kernelBandwidth = 2.0f * 1000.0f * mem_size / (1024 * 1024 * 1024) /
                            (kernelTime / NTIMES);
    printf("Effective throughput = %.4f GB/s\n", kernelBandwidth);

    checkCudaErrors(cudaMemcpy(ans_h, matA_d, sizeof(float) * NX * NY,
                               cudaMemcpyDeviceToHost));

    if (checkAnswer(ans_h, matB_h, NX, NY)) {
      std::cout << "right answer" << "\n";
    } else {
      std::cout << "wrong answer" << "\n";
    }
  }

  checkCudaErrors(cudaFree(matA_d));
  checkCudaErrors(cudaFree(matB_d));
  free(matA_h);
  free(matB_h);
}
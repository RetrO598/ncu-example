#pragma once
#include <curand_mtgp32_kernel.h>
__global__ void matrixTransposeNaive(float *matA, const float *matB,
                                     const int NX, const int NY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < NX && j < NY) {
    matA[i * NY + j] = matB[j * NX + i];
  }
}

__global__ void matrixTranposeShared(float *matA, const float *matB,
                                     const int NX, const int NY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  __shared__ float tmp[32][32];
  if (i < NX && j < NY) {
    int x = blockDim.y * blockIdx.y + threadIdx.x;
    int y = blockDim.x * blockIdx.x + threadIdx.y;
    tmp[threadIdx.x][threadIdx.y] = matB[j * NX + i];
    __syncthreads();
    if (x < NY && y < NX) {
      matA[y * NY + x] = tmp[threadIdx.y][threadIdx.x];
    }
  }
}

__global__ void matrixTranposeSharedPadding(float *matA, const float *matB,
                                            const int NX, const int NY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  __shared__ float tmp[32][33];
  if (i < NX && j < NY) {
    int x = blockDim.y * blockIdx.y + threadIdx.x;
    int y = blockDim.x * blockIdx.x + threadIdx.y;
    tmp[threadIdx.x][threadIdx.y] = matB[j * NX + i];
    __syncthreads();
    if (x < NY && y < NX) {
      matA[y * NY + x] = tmp[threadIdx.y][threadIdx.x];
    }
  }
}

__global__ void matrixTranposeSharedSwizz(float *matA, const float *matB,
                                          const int NX, const int NY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  __shared__ float tmp[32][33];
  if (i < NX && j < NY) {
    int x = blockDim.y * blockIdx.y + threadIdx.x;
    int y = blockDim.x * blockIdx.x + threadIdx.y;
    tmp[threadIdx.x][(threadIdx.x + threadIdx.y) % 32] = matB[j * NX + i];
    __syncthreads();
    if (x < NY && y < NX) {
      matA[y * NY + x] = tmp[threadIdx.y][(threadIdx.x + threadIdx.y) % 2];
    }
  }
}
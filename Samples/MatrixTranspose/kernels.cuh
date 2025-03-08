#pragma once

template <int BLOCK_SIZE>
__global__ void matrixCopyShared(float *matA, const float *matB, const int NX,
                                 const int NY) {
  __shared__ float tmp[BLOCK_SIZE][BLOCK_SIZE];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < NX && j < NY) {
    tmp[threadIdx.y][threadIdx.x] = matB[j * NX + i];

    __syncthreads();
    matA[j * NX + i] = tmp[threadIdx.y][threadIdx.x];
  }
}

__global__ void matrixTransposeNaive(float *matA, const float *matB,
                                     const int NX, const int NY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < NX && j < NY) {
    matA[i * NY + j] = matB[j * NX + i];
  }
}

template <int BLOCK_SIZE>
__global__ void matrixTransposeShared(float *matA, const float *matB,
                                      const int NX, const int NY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  __shared__ float tmp[BLOCK_SIZE][BLOCK_SIZE];
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

template <int BLOCK_SIZE>
__global__ void matrixTransposeSharedPadding(float *matA, const float *matB,
                                             const int NX, const int NY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  __shared__ float tmp[BLOCK_SIZE][BLOCK_SIZE + 1];
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

template <int BLOCK_SIZE>
__global__ void matrixTransposeSharedSwizz(float *matA, const float *matB,
                                           const int NX, const int NY) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  __shared__ float tmp[BLOCK_SIZE][BLOCK_SIZE];
  if (i < NX && j < NY) {
    int x = blockDim.y * blockIdx.y + threadIdx.x;
    int y = blockDim.x * blockIdx.x + threadIdx.y;
    tmp[threadIdx.x][(threadIdx.x + threadIdx.y) % BLOCK_SIZE] =
        matB[j * NX + i];
    __syncthreads();
    if (x < NY && y < NX) {
      matA[y * NY + x] =
          tmp[threadIdx.y][(threadIdx.x + threadIdx.y) % BLOCK_SIZE];
    }
  }
}
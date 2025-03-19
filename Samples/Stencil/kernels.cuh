#pragma once

// Stencil constants
constexpr float c0 = 0.5f;
constexpr float c1 = 0.5f;
constexpr float c2 = 0.5f;
constexpr float c3 = 0.5f;
constexpr float c4 = 0.5f;
constexpr float c5 = 0.5f;
constexpr float c6 = 0.5f;

constexpr int BLOCK_SIZE = 8;
constexpr int OUT_TILE = BLOCK_SIZE - 2;

__global__ void stencilNaive(float *in, float *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i + j * N + k * N * N] = c0 * in[i + j * N + k * N * N] +
                                 c1 * in[(i - 1) + j * N + k * N * N] +
                                 c2 * in[(i + 1) + j * N + k * N * N] +
                                 c3 * in[i + (j - 1) * N + k * N * N] +
                                 c4 * in[i + (j + 1) * N + k * N * N] +
                                 c5 * in[i + j * N + (k - 1) * N * N] +
                                 c6 * in[i + j * N + (k + 1) * N * N];
  }
}

__global__ void stencilShared(float *in, float *out, int N) {
  __shared__ float TILE[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

  int global_x = blockIdx.x * OUT_TILE + threadIdx.x - 1;
  int global_y = blockIdx.y * OUT_TILE + threadIdx.y - 1;
  int global_z = blockIdx.z * OUT_TILE + threadIdx.z - 1;

  if (global_x >= 0 && global_x < N && global_y >= 0 && global_y < N &&
      global_z >= 0 && global_z < N) {
    TILE[threadIdx.z][threadIdx.y][threadIdx.x] =
        in[global_x + global_y * N + global_z * N * N];
  }

  __syncthreads();

  int local_x = threadIdx.x;
  int local_y = threadIdx.y;
  int local_z = threadIdx.z;

  if (global_x > 0 && global_x < N - 1 && global_y > 0 && global_y < N - 1 &&
      global_z > 0 && global_z < N - 1) {
    if (local_x >= 1 && local_x < BLOCK_SIZE - 1 && local_y >= 1 &&
        local_y < BLOCK_SIZE - 1 && local_z >= 1 && local_z < BLOCK_SIZE - 1) {
      out[global_x + global_y * N + global_z * N * N] =
          c0 * TILE[local_z][local_y][local_x] +
          c1 * TILE[local_z][local_y][local_x - 1] +
          c2 * TILE[local_z][local_y][local_x + 1] +
          c3 * TILE[local_z][local_y - 1][local_x] +
          c4 * TILE[local_z][local_y + 1][local_x] +
          c5 * TILE[local_z - 1][local_y][local_x] +
          c6 * TILE[local_z + 1][local_y][local_x];
    }
  }
}

template <int blocksize, int output_tile>
__global__ void stencilThreadCoarsen(float *in, float *out, int N) {
  int k = blockIdx.z * output_tile;
  int j = blockIdx.y * output_tile + threadIdx.y - 1;
  int i = blockIdx.x * output_tile + threadIdx.x - 1;

  __shared__ float inPrev_s[blocksize][blocksize];
  __shared__ float inCurr_s[blocksize][blocksize];
  __shared__ float inNext_s[blocksize][blocksize];

  if (i >= 0 && i < N && j >= 0 && j < N && k - 1 >= 0 && k - 1 < N) {
    inPrev_s[threadIdx.y][threadIdx.x] = in[i + j * N + (k - 1) * N * N];
  }

  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    inCurr_s[threadIdx.y][threadIdx.x] = in[i + j * N + k * N * N];
  }

  for (int l = k; l < k + OUT_TILE; ++l) {
    if (l + 1 >= 0 && l + 1 < N && i >= 0 && i < N && j >= 0 && j < N) {
      inNext_s[threadIdx.y][threadIdx.x] = in[i + j * N + (l + 1) * N * N];
    }

    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && l >= 1 && l < N - 1) {
      if (threadIdx.x >= 1 && threadIdx.x < blocksize - 1 && threadIdx.y >= 1 &&
          threadIdx.y < blocksize - 1) {
        out[i + j * N + l * N * N] =
            c0 * inCurr_s[threadIdx.y][threadIdx.x] +
            c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
            c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
            c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
            c4 * inCurr_s[threadIdx.y + 1][threadIdx.x] +
            c5 * inPrev_s[threadIdx.y][threadIdx.x] +
            c6 * inNext_s[threadIdx.y][threadIdx.x];
      }
    }
    __syncthreads();
    inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
    inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
  }
}

template <int blocksize, int output_tile>
__global__ void stencilThreadCoarsenRegister(float *in, float_t *out, int N) {
  int k = blockIdx.z * output_tile;
  int j = blockIdx.y * output_tile + threadIdx.y - 1;
  int i = blockIdx.x * output_tile + threadIdx.x - 1;

  __shared__ float inCurr_s[blocksize][blocksize];

  float inPrev;
  float inNext;
  float inCurr;

  if (i >= 0 && i < N && j >= 0 && j < N && k - 1 >= 0 && k - 1 < N) {
    inPrev = in[i + j * N + (k - 1) * N * N];
  }
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    inCurr = in[i + j * N + k * N * N];
    inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
  }

  for (int l = k; l < k + OUT_TILE; ++l) {
    if (l + 1 >= 0 && l + 1 < N && i >= 0 && i < N && j >= 0 && j < N) {
      inNext = in[i + j * N + (l + 1) * N * N];
    }

    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && l >= 1 && l < N - 1) {
      if (threadIdx.x >= 1 && threadIdx.x < blocksize - 1 && threadIdx.y >= 1 &&
          threadIdx.y < blocksize - 1) {
        out[i + j * N + l * N * N] =
            c0 * inCurr_s[threadIdx.y][threadIdx.x] +
            c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
            c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
            c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
            c4 * inCurr_s[threadIdx.y + 1][threadIdx.x] + c5 * inPrev +
            c6 * inNext;
      }
    }
    __syncthreads();
    inPrev = inCurr;
    inCurr = inNext;
    inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
  }
}
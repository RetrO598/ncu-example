#pragma once
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

__global__ void sgemmNaive(float *A, float *B, float *C, int M, int N, int K) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;

  if (tx < K && ty < M) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
      sum += A[i + ty * N] * B[tx + i * K];
    }
    C[tx + ty * K] = sum;
  }
}

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N, int SHARED_M, int SHARED_K, int SHARED_N>
__global__ void sgemmShared_v1(float *__restrict__ A, float *__restrict__ B,
                               float *__restrict__ C, int M, int N, int K) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float a_shared[SHARED_M][SHARED_K];
  __shared__ float b_shared[SHARED_K][SHARED_N];

  float r_c[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

  int load_a_smem_m = tid / (SHARED_K / 4);
  int load_a_smem_k = (tid % (SHARED_K / 4)) * 4;
  int load_b_smem_k = tid / (SHARED_N / 4);
  int load_b_smem_n = (tid % (SHARED_N / 4)) * 4;

  int load_a_gmem_m = by * BLOCK_TILE_M + load_a_smem_m;
  int load_b_gmem_n = bx * BLOCK_TILE_N + load_b_smem_n;

#pragma unroll
  for (int k = 0; k < K; k += SHARED_K) {
    int load_a_gmem_k = load_a_smem_k + k;
    int load_b_gmem_k = load_b_smem_k + k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

    FLOAT4(a_shared[load_a_smem_m][load_a_smem_k]) =
        FLOAT4(A[load_a_gmem_addr]);
    FLOAT4(b_shared[load_b_smem_k][load_b_smem_n]) =
        FLOAT4(B[load_b_gmem_addr]);

    __syncthreads();

#pragma unroll
    for (int m = 0; m < THREAD_TILE_M; ++m) {
#pragma unroll
      for (int n = 0; n < THREAD_TILE_N; ++n) {
#pragma unroll
        for (int i = 0; i < SHARED_K; ++i) {
          r_c[m][n] += a_shared[ty * THREAD_TILE_M + m][i] *
                       b_shared[i][tx * THREAD_TILE_N + n];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < THREAD_TILE_M; ++m) {
#pragma unroll
    for (int n = 0; n < THREAD_TILE_N; n += 4) {
      int store_c_gmem_m = by * BLOCK_TILE_M + ty * THREAD_TILE_M + m;
      int store_c_gmem_n = bx * BLOCK_TILE_N + tx * THREAD_TILE_N + n;
      int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
      FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N, int SHARED_M, int SHARED_K, int SHARED_N>
__global__ void sgemmShared_v2(float *__restrict__ A, float *__restrict__ B,
                               float *__restrict__ C, int M, int N, int K) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float a_shared[SHARED_M][SHARED_K];
  __shared__ float b_shared[SHARED_K][SHARED_N];

  float r_c[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
  float r_a[THREAD_TILE_M] = {0.0f};
  float r_b[THREAD_TILE_N] = {0.0f};

  int load_a_smem_m = tid / (SHARED_K / 4);
  int load_a_smem_k = (tid % (SHARED_K / 4)) * 4;
  int load_b_smem_k = tid / (SHARED_N / 4);
  int load_b_smem_n = (tid % (SHARED_N / 4)) * 4;

  int load_a_gmem_m = by * BLOCK_TILE_M + load_a_smem_m;
  int load_b_gmem_n = bx * BLOCK_TILE_N + load_b_smem_n;

#pragma unroll
  for (int k = 0; k < K; k += SHARED_K) {
    int load_a_gmem_k = load_a_smem_k + k;
    int load_b_gmem_k = load_b_smem_k + k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

    FLOAT4(a_shared[load_a_smem_m][load_a_smem_k]) =
        FLOAT4(A[load_a_gmem_addr]);
    FLOAT4(b_shared[load_b_smem_k][load_b_smem_n]) =
        FLOAT4(B[load_b_gmem_addr]);

    __syncthreads();

#pragma unroll
    for (int i = 0; i < SHARED_K; ++i) {
      for (int m = 0; m < THREAD_TILE_M; ++m) {
        r_a[m] = a_shared[ty * THREAD_TILE_M + m][i];
      }
      for (int n = 0; n < THREAD_TILE_N; n += 4) {
        FLOAT4(r_b[n]) = FLOAT4(b_shared[i][tx * THREAD_TILE_N + n]);
      }
#pragma unroll
      for (int m = 0; m < THREAD_TILE_M; ++m) {
#pragma unroll
        for (int n = 0; n < THREAD_TILE_N; ++n) {
          r_c[m][n] += r_a[m] * r_b[n];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < THREAD_TILE_M; ++m) {
#pragma unroll
    for (int n = 0; n < THREAD_TILE_N; n += 4) {
      int store_c_gmem_m = by * BLOCK_TILE_M + ty * THREAD_TILE_M + m;
      int store_c_gmem_n = bx * BLOCK_TILE_N + tx * THREAD_TILE_N + n;
      int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
      FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N, int SHARED_M, int SHARED_K, int SHARED_N>
__global__ void sgemmShared_v3(float *__restrict__ A, float *__restrict__ B,
                               float *__restrict__ C, int M, int N, int K) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float a_shared[SHARED_K][SHARED_M];
  __shared__ float b_shared[SHARED_K][SHARED_N];

  float r_c[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
  float r_a[THREAD_TILE_M] = {0.0f};
  float r_b[THREAD_TILE_N] = {0.0f};

  int load_a_smem_m = tid / (SHARED_K / 4);
  int load_a_smem_k = (tid % (SHARED_K / 4)) * 4;
  int load_b_smem_k = tid / (SHARED_N / 4);
  int load_b_smem_n = (tid % (SHARED_N / 4)) * 4;

  int load_a_gmem_m = by * BLOCK_TILE_M + load_a_smem_m;
  int load_b_gmem_n = bx * BLOCK_TILE_N + load_b_smem_n;

#pragma unroll
  for (int k = 0; k < K; k += SHARED_K) {
    int load_a_gmem_k = load_a_smem_k + k;
    int load_b_gmem_k = load_b_smem_k + k;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
    float trans_a[4] = {0.0f};
    FLOAT4(trans_a[0]) = FLOAT4(A[load_a_gmem_addr]);
    for (int i = 0; i < 4; ++i) {
      a_shared[load_a_smem_k + i][load_a_smem_m] = trans_a[i];
    }
    FLOAT4(b_shared[load_b_smem_k][load_b_smem_n]) =
        FLOAT4(B[load_b_gmem_addr]);

    __syncthreads();

#pragma unroll
    for (int i = 0; i < SHARED_K; ++i) {
      for (int m = 0; m < THREAD_TILE_M; m += 4) {
        // r_a[m] = a_shared[ty * THREAD_TILE_M + m][i];
        FLOAT4(r_a[m]) = FLOAT4(a_shared[i][ty * THREAD_TILE_M + m]);
      }
      for (int n = 0; n < THREAD_TILE_N; n += 4) {
        FLOAT4(r_b[n]) = FLOAT4(b_shared[i][tx * THREAD_TILE_N + n]);
      }
#pragma unroll
      for (int m = 0; m < THREAD_TILE_M; ++m) {
#pragma unroll
        for (int n = 0; n < THREAD_TILE_N; ++n) {
          r_c[m][n] += r_a[m] * r_b[n];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < THREAD_TILE_M; ++m) {
#pragma unroll
    for (int n = 0; n < THREAD_TILE_N; n += 4) {
      int store_c_gmem_m = by * BLOCK_TILE_M + ty * THREAD_TILE_M + m;
      int store_c_gmem_n = bx * BLOCK_TILE_N + tx * THREAD_TILE_N + n;
      int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
      FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}
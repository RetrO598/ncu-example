#include "kernels.cuh"
#include <cstdio>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>
#include <helper_functions.h>

void cpuSgemm(const float *A, const float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

float testMaxError(void (*gpuSgemm)(float *, float *, float *, const int,
                                    const int, const int),
                   dim3 gridDim, dim3 blockDim, const int M, const int N,
                   const int K) {

  size_t size_a = M * K * sizeof(float);
  size_t size_b = K * N * sizeof(float);
  size_t size_c = M * N * sizeof(float);

  float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
  h_a = (float *)malloc(size_a);
  h_b = (float *)malloc(size_b);
  h_c = (float *)malloc(size_c);
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);
  h_d_c = (float *)malloc(size_c);

  srand(time(0));
  for (int i = 0; i < M * K; i++)
    h_a[i] = rand() / float(RAND_MAX);
  for (int i = 0; i < K * N; i++)
    h_b[i] = rand() / float(RAND_MAX);
  cudaMemset(d_c, 15, size_c);

  cpuSgemm(h_a, h_b, h_c, M, N, K);

  cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
  gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
  cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

  float max_error = 0.0;
  for (int i = 0; i < M * N; i++) {
    float this_error = abs(h_d_c[i] - h_c[i]);
    if (max_error != max_error || this_error != this_error) // nan
      max_error = -NAN;
    else
      max_error = std::max(max_error, this_error);
  }

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_d_c);

  return max_error;
}

float testPerformance(void (*gpuSgemm)(float *, float *, float *, const int,
                                       const int, const int),
                      dim3 gridDim, dim3 blockDim, const int M, const int N,
                      const int K, const int repeat) {

  size_t size_a = M * K * sizeof(float);
  size_t size_b = K * N * sizeof(float);
  size_t size_c = M * N * sizeof(float);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++)
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec, sec;
  cudaEventElapsedTime(&msec, start, end);
  sec = msec / 1000.0 / repeat;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return sec;
}

int main() {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  constexpr int BLOCK_TILE_M = 128;
  constexpr int BLOCK_TILE_N = 128;
  constexpr int THREAD_TILE_M = 8;
  constexpr int THREAD_TILE_N = 8;
  constexpr int SHARED_M = 128;
  constexpr int SHARED_K = 8;
  constexpr int SHARED_N = 128;
  dim3 blockDim(BLOCK_TILE_N / THREAD_TILE_N, BLOCK_TILE_M / THREAD_TILE_M);
  dim3 gridDim((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
               (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  dim3 blockDim2(16, 16);
  dim3 gridDim2((N + 16 - 1) / 16, (M + 16 - 1) / 16);

  double total_sec = 0.0;
  for (int i = 0; i < 10; ++i) {
    double this_sec = testPerformance(
        sgemmShared_v3<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N,
                       SHARED_M, SHARED_K, SHARED_N>,
        gridDim, blockDim, M, N, K, 1);

    total_sec += this_sec;
  }
  double ave_sec = total_sec / 10.0;
  double ave_gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / ave_sec;

  printf("Average time = %f sec, GFLOPS = %f\n", ave_sec, ave_gflops);
  float max_error =
      testMaxError(sgemmShared_v3<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M,
                                  THREAD_TILE_N, SHARED_M, SHARED_K, SHARED_N>,
                   gridDim, blockDim, M, N, K);
  printf("Max Error = %f\n", max_error);
  return 0;
}
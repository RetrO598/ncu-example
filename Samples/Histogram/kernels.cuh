#pragma once

constexpr int HISTO_BIN = 26;
constexpr int BLOCK_SIZE = 256;

__global__ void histogramNaive(char *data, unsigned int length,
                               unsigned int *histo) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int position = data[i] - 'a';
    if (position >= 0 && position < 26) {
      atomicAdd(&histo[position], 1);
    }
  }
}

__global__ void histogramPrivatization(char *data, unsigned int length,
                                       unsigned int *histo) {
  __shared__ unsigned int histo_private[HISTO_BIN];
  for (int i = threadIdx.x; i < HISTO_BIN; i += blockDim.x) {
    histo_private[i] = 0;
  }

  __syncthreads();

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) {
    int position = data[i] - 'a';
    if (position >= 0 && position < 26) {
      atomicAdd(&histo_private[position], 1);
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < HISTO_BIN; i += blockDim.x) {
    atomicAdd(&histo[i], histo_private[i]);
  }
}

template <int COARSEN_FACTOR>
__global__ void histogramCoarsenContiguous(char *data, unsigned int length,
                                           unsigned int *histo) {
  __shared__ unsigned int histo_private[HISTO_BIN];
  for (int i = threadIdx.x; i < HISTO_BIN; i += blockDim.x) {
    histo_private[i] = 0;
  }

  __syncthreads();

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int i = idx * COARSEN_FACTOR; i < (idx + 1) * COARSEN_FACTOR;
       ++i) {
    if (i < length) {
      int position = data[i] - 'a';
      if (position >= 0 && position < 26) {
        atomicAdd(&histo_private[position], 1);
      }
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < HISTO_BIN; i += blockDim.x) {
    atomicAdd(&histo[i], histo_private[i]);
  }
}

__global__ void histogramCoarsenInterleaved(char *data, unsigned int length,
                                            unsigned int *histo) {
  __shared__ unsigned int histo_private[HISTO_BIN];
  for (int i = threadIdx.x; i < HISTO_BIN; i += blockDim.x) {
    histo_private[i] = 0;
  }

  __syncthreads();

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int i = idx; i < length; i += blockDim.x * gridDim.x) {
    int position = data[i] - 'a';
    if (position >= 0 && position < 26) {
      atomicAdd(&histo_private[position], 1);
    }
  }

  __syncthreads();
  for (int i = threadIdx.x; i < HISTO_BIN; i += blockDim.x) {
    atomicAdd(&histo[i], histo_private[i]);
  }
}

__global__ void histogramAggregation(char *data, unsigned int length,
                                     unsigned int *histo) {
  __shared__ unsigned int histo_private[HISTO_BIN];
  for (int i = threadIdx.x; i < HISTO_BIN; i += blockDim.x) {
    histo_private[i] = 0;
  }

  __syncthreads();

  unsigned int accumulators = 0;
  int prevBinIdx = -1;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int i = idx; i < length; i += blockDim.x * gridDim.x) {
    int position = data[i] - 'a';
    if (position >= 0 && position < 26) {
      if (position == prevBinIdx) {
        accumulators++;
      } else {
        if (prevBinIdx != -1) {
          atomicAdd(&histo_private[prevBinIdx], accumulators);
        }
        prevBinIdx = position;
        accumulators = 1;
      }
    }
  }

  if (accumulators > 0) {
    atomicAdd(&histo_private[prevBinIdx], accumulators);
  }

  __syncthreads();

  for (int i = threadIdx.x; i < HISTO_BIN; i += blockDim.x) {
    atomicAdd(&histo[i], histo_private[i]);
  }
}
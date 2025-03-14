#pragma once

constexpr int filterRadius = 1;
constexpr int BLOCKSIZE = 32;
constexpr int INPUT_TILE = BLOCKSIZE;
constexpr int OUTPUT_TILE = BLOCKSIZE - 2 * filterRadius;
__constant__ float constFilter[2 * filterRadius + 1][2 * filterRadius + 1];
__global__ void convolutionNaive(float *input, float *filter, float *output,
                                 int width, int height) {

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  float value = 0.0f;
  int xinput, yinput;
  for (int j = 0; j < 2 * filterRadius + 1; ++j) {
    for (int i = 0; i < 2 * filterRadius + 1; ++i) {
      xinput = x - filterRadius + i;
      yinput = y - filterRadius + j;
      if (xinput >= 0 && xinput < width && yinput >= 0 && yinput < height) {
        value += input[yinput * width + xinput] *
                 filter[j * (2 * filterRadius + 1) + i];
      }
    }
  }

  output[y * width + x] = value;
}

__global__ void convolutionConst(float *input, float *output, int width,
                                 int height) {

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  float value = 0.0f;
  int xinput, yinput;
  for (int j = 0; j < 2 * filterRadius + 1; ++j) {
    for (int i = 0; i < 2 * filterRadius + 1; ++i) {
      xinput = x - filterRadius + i;
      yinput = y - filterRadius + j;
      if (xinput >= 0 && xinput < width && yinput >= 0 && yinput < height) {
        value += input[yinput * width + xinput] * constFilter[j][i];
      }
    }
  }

  output[y * width + x] = value;
}

// The size of shared memory equals blocksize, which including output_tile and
// halo, the grid size should be calculated through output_tile instead of
__global__ void convolutionSharedHalo(float *input, float *output, int width,
                                      int height) {
  __shared__ float inTile[INPUT_TILE][INPUT_TILE];

  int x = blockIdx.x * OUTPUT_TILE + threadIdx.x - filterRadius;
  int y = blockIdx.y * OUTPUT_TILE + threadIdx.y - filterRadius;

  if (x >= 0 && x < width && y >= 0 && y < height) {
    inTile[threadIdx.y][threadIdx.x] = input[y * width + x];
  } else {
    inTile[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();
  int i = threadIdx.x - filterRadius;
  int j = threadIdx.y - filterRadius;
  if (i >= 0 && i < OUTPUT_TILE && j >= 0 && j < OUTPUT_TILE) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
      float value = 0.0f;
      for (int m = 0; m < 2 * filterRadius + 1; ++m) {
        for (int n = 0; n < 2 * filterRadius + 1; ++n) {
          value += constFilter[m][n] * inTile[j + m][i + n];
        }
      }
      output[y * width + x] = value;
    }
  }
}

__global__ void convolutionShared(float *input, float *output, int width,
                                  int height) {
  int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

  __shared__ float inTile[BLOCKSIZE][BLOCKSIZE];
  if (x < width && y < height) {
    inTile[threadIdx.y][threadIdx.x] = input[y * width + x];
  } else {
    inTile[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();
  if (x < width && y < height) {
    float value = 0.0f;
    for (int j = 0; j < 2 * filterRadius + 1; ++j) {
      for (int i = 0; i < 2 * filterRadius + 1; ++i) {
        int inputx = threadIdx.x - filterRadius + i;
        int inputy = threadIdx.y - filterRadius + j;
        if (inputx >= 0 && inputx < BLOCKSIZE && inputy >= 0 &&
            inputy < BLOCKSIZE) {
          value += constFilter[j][i] * inTile[inputy][inputx];
        } else if (x - filterRadius + i >= 0 && x - filterRadius + i < width &&
                   y - filterRadius + j >= 0 && y - filterRadius + j < height) {
          value +=
              constFilter[j][i] *
              input[(y - filterRadius + j) * width + (x - filterRadius + i)];
        }
      }
    }
    output[y * width + x] = value;
  }
}
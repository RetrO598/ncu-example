#pragma once

__global__ void convolutionNaive(float *input, float *filter, float *output,
                                 int radius, int width, int height) {

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  float value = 0.0f;
  int xinput, yinput;
  for (int j = 0; j < 2 * radius + 1; ++j) {
    for (int i = 0; i < 2 * radius + 1; ++i) {
      xinput = x - radius + i;
      yinput = y - radius + j;
      if (xinput >= 0 && xinput < width && yinput >= 0 && yinput < height) {
        value +=
            input[yinput * width + xinput] * filter[j * (2 * radius + 1) + i];
      }
    }
  }

  output[y * width + x] = value;
}

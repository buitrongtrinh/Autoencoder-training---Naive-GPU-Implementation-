// kernels.cuh
#pragma once

#include <cuda_runtime.h>

// Convolution (naive, có batch)
__global__ void conv_naive(
    float* input,    // N x H x W x Cin
    float* weight,   // Cout x Cin x 3 x 3
    float* bias,     // Cout
    float* output,   // N x H x W x Cout
    int H, int W, int Cin, int Cout
);

// ReLU
__global__ void relu_naive(
    float* x,
    int n
);

// MaxPooling 2x2
__global__ void maxpool2x2(
    float* input,    // N x H x W x C
    float* output,   // N x H/2 x W/2 x C
    int H, int W, int C
);

// Upsampling 2x
__global__ void upsample2x_naive(
    float* input,    // N x H x W x C
    float* output,   // N x 2H x 2W x C
    int H, int W, int C
);

// MSE Loss
__global__ void mse_loss_kernel(
    float* pred,
    float* target,
    float* loss,
    int n
);

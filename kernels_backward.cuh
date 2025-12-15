// kernels_backward.cuh
#pragma once
#include <cuda_runtime.h>

// ===== MSE backward =====
__global__ void mse_backward(
    float* pred,
    float* target,
    float* d_pred,
    int n
);

// ===== ReLU backward =====
__global__ void relu_backward(
    float* input,
    float* d_out,
    float* d_in,
    int n
);

// ===== MaxPool 2x2 backward =====
__global__ void maxpool_backward(
    float* input,
    float* d_out,
    float* d_in,
    int H, int W, int C
);

// ===== Upsample backward (nearest neighbor) =====
__global__ void upsample_backward(
    float* d_out,
    float* d_in,
    int H, int W, int C
);

// ===== Conv weight backward =====
__global__ void conv_weight_backward(
    float* input,
    float* d_out,
    float* d_weight,
    int H, int W, int Cin, int Cout, int batch_size
);
// ===== Conv bias backward =====
__global__ void conv_bias_backward(
    float* d_out,
    float* d_bias,
    int H, int W, int Cout, int batch_size
);
// Conv input gradient
__global__ void conv_input_backward(
    float* d_out,
    float* weight,
    float* d_in,
    int H, int W, int Cin, int Cout
);
// ===== SGD update =====
__global__ void sgd_update(
    float* weight,
    float* grad,
    float lr,
    int n
);

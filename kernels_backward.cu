// kernels_backward.cu
#include "kernels_backward.cuh"
#include <math.h>

// ================= MSE backward =================
__global__ void mse_backward(
    float* pred,
    float* target,
    float* d_pred,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_pred[idx] = 2.0f * (pred[idx] - target[idx]) / n;
    }
}

// ================= ReLU backward =================
__global__ void relu_backward(
    float* input,
    float* d_out,
    float* d_in,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_in[idx] = (input[idx] > 0) ? d_out[idx] : 0.0f;
    }
}

// ================= MaxPool backward =================
__global__ void maxpool_backward(
    float* input,
    float* d_out,
    float* d_in,
    int H, int W, int C
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int bz = blockIdx.z;
    int n  = bz / C;
    int c  = bz % C;

    int outW = W / 2;
    int outH = H / 2;
    if (x >= outW || y >= outH) return;

    int ix = x * 2;
    int iy = y * 2;

    float maxv = -1e20f;
    int max_idx = -1;

    for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++) {
        int idx = ((n*H + iy+dy)*W + ix+dx)*C + c;
        if (input[idx] > maxv) {
            maxv = input[idx];
            max_idx = idx;
        }
    }

    int out_idx = ((n*outH + y)*outW + x)*C + c;
    atomicAdd(&d_in[max_idx], d_out[out_idx]);
}

// ================= Upsample backward =================
__global__ void upsample_backward(
    float* d_out,
    float* d_in,
    int H, int W, int C
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int bz = blockIdx.z;
    int n  = bz / C;
    int c  = bz % C;

    int outH = H * 2;
    int outW = W * 2;
    if (x >= outW || y >= outH) return;

    int in_x = x >> 1;
    int in_y = y >> 1;

    int out_idx = ((n*outH + y)*outW + x)*C + c;
    int in_idx  = ((n*H + in_y)*W + in_x)*C + c;

    atomicAdd(&d_in[in_idx], d_out[out_idx]);
}

// ================= Conv weight backward =================
__global__ void conv_weight_backward(
    float* input,
    float* d_out,
    float* d_weight,
    int H, int W, int Cin, int Cout, int batch_size
) {
    int oc = blockIdx.x;
    int c  = blockIdx.y;
    int ky = threadIdx.y;
    int kx = threadIdx.x;

    float sum = 0.0f;

    for (int n = 0; n < batch_size; n++)
    for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
        int ix = x + kx - 1;
        int iy = y + ky - 1;
        if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
            int in_idx  = ((n*H + iy)*W + ix)*Cin + c;
            int out_idx = ((n*H + y)*W + x)*Cout + oc;
            sum += input[in_idx] * d_out[out_idx];
        }
    }

    int w_idx = ((oc*Cin + c)*3 + ky)*3 + kx;
    atomicAdd(&d_weight[w_idx], sum);
}
// ================= Conv bias backward =================
__global__ void conv_bias_backward(
    float* d_out,
    float* d_bias,
    int H, int W, int Cout,
    int batch_size  // <-- THÊM PARAMETER NÀY
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= Cout) return;

    float sum = 0.0f;

    // ✅ SỬA: Loop qua tất cả batches
    for (int n = 0; n < batch_size; n++)  // <-- ĐỔI TỪ 1 THÀNH batch_size
    for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
        int out_idx = ((n*H + y)*W + x)*Cout + oc;
        sum += d_out[out_idx];
    }

    atomicAdd(&d_bias[oc], sum);  // <-- ĐỔI THÀNH atomicAdd
}
// ================= Conv input backward =================
__global__ void conv_input_backward(
    float* d_out,      // gradient from output: (H, W, Cout)
    float* weight,     // conv weights: (Cout, Cin, 3, 3)
    float* d_in,       // gradient to input: (H, W, Cin)
    int H, int W, int Cin, int Cout
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int bz = blockIdx.z;
    int n  = bz / Cin;
    int c  = bz % Cin;
    
    if (x >= W || y >= H) return;
    
    float sum = 0.0f;
    
    // For each output channel
    for (int oc = 0; oc < Cout; oc++) {
        // For each position in 3x3 kernel
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                // Output position that uses this input
                int ox = x - kx + 1;  // padding=1, so shift by (kx-1)
                int oy = y - ky + 1;
                
                if (ox >= 0 && ox < W && oy >= 0 && oy < H) {
                    int out_idx = ((n*H + oy)*W + ox)*Cout + oc;
                    int w_idx = ((oc*Cin + c)*3 + ky)*3 + kx;
                    
                    sum += d_out[out_idx] * weight[w_idx];
                }
            }
        }
    }
    
    int in_idx = ((n*H + y)*W + x)*Cin + c;
    d_in[in_idx] = sum;
}
// ================= SGD update =================
__global__ void sgd_update(
    float* weight,
    float* grad,
    float lr,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weight[idx] -= lr * grad[idx];
    }
}


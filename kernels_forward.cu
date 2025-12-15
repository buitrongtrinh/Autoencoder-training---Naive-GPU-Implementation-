// kernels.cu
#include "kernels_forward.cuh"
#include <math.h>

__global__ void conv_naive(
    float* input,
    float* weight,
    float* bias,
    float* output,
    int H, int W, int Cin, int Cout
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int bz = blockIdx.z;
    int n  = bz / Cout;
    int oc = bz % Cout;

    if (x >= W || y >= H) return;

    float sum = bias[oc];

    for (int ky = -1; ky <= 1; ky++)
    for (int kx = -1; kx <= 1; kx++)
    for (int c = 0; c < Cin; c++) {

        int ix = x + kx;
        int iy = y + ky;

        if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
            int in_idx =
                ((n * H + iy) * W + ix) * Cin + c;

            int w_idx =
                ((oc * Cin + c) * 3 + (ky + 1)) * 3 + (kx + 1);

            sum += input[in_idx] * weight[w_idx];
        }
    }

    int out_idx =
        ((n * H + y) * W + x) * Cout + oc;

    output[out_idx] = sum;
}

__global__ void relu_naive(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void maxpool2x2(
    float* input,
    float* output,
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

    float m = -1e20f;
    for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++) {
        int in_idx =
            ((n * H + (iy + dy)) * W + (ix + dx)) * C + c;
        m = fmaxf(m, input[in_idx]);
    }

    int out_idx =
        ((n * outH + y) * outW + x) * C + c;

    output[out_idx] = m;
}

__global__ void upsample2x_naive(
    float* input,
    float* output,
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

    int in_idx =
        ((n * H + in_y) * W + in_x) * C + c;

    int out_idx =
        ((n * outH + y) * outW + x) * C + c;

    output[out_idx] = input[in_idx];
}

__global__ void mse_loss_kernel(
    float* pred,
    float* target,
    float* loss,
    int n
) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (idx < n) {
        float diff = pred[idx] - target[idx];
        val = diff * diff;
    }

    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(loss, sdata[0]);
}

#include "gpu_AE.cuh"
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include "kernels_forward.cuh"
#include "kernels_backward.cuh"

#define CUDA_CHECK(call)                                      \
do {                                                         \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                 \
                  << " at " << __FILE__ << ":" << __LINE__   \
                  << std::endl;                              \
        exit(1);                                             \
    }                                                        \
} while (0)

gpu_AE::gpu_AE(int batch_size) : batch_size(batch_size) {

    // -------- WEIGHTS --------
    CUDA_CHECK(cudaMalloc(&w1, 3 * 3 * 3 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w2, 3 * 3 * 256 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w3, 3 * 3 * 128 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w4, 3 * 3 * 128 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w5, 3 * 3 * 256 * 3 * sizeof(float)));

    // -------- BIAS --------
    CUDA_CHECK(cudaMalloc(&b1, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b2, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b3, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b4, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b5, 3   * sizeof(float)));

    // -------- FORWARD OUTPUTS --------
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * 32 * 32 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o1, batch_size * 32 * 32 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o2, batch_size * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o3, batch_size * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o4, batch_size * 8  * 8  * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o5, batch_size * 8  * 8  * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o6, batch_size * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o7, batch_size * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o8, batch_size * 32 * 32 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&o9, batch_size * 32 * 32 * 3   * sizeof(float)));

    // -------- BACKWARD GRADIENT BUFFERS --------
    CUDA_CHECK(cudaMalloc(&d_o1, batch_size * 32 * 32 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o2, batch_size * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o3, batch_size * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o4, batch_size * 8  * 8  * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o5, batch_size * 8  * 8  * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o6, batch_size * 16 * 16 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o7, batch_size * 16 * 16 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o8, batch_size * 32 * 32 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o9, batch_size * 32 * 32 * 3   * sizeof(float)));

    // -------- WEIGHT GRADIENTS --------
    CUDA_CHECK(cudaMalloc(&dw1, 3 * 3 * 3   * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw2, 3 * 3 * 256 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw3, 3 * 3 * 128 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw4, 3 * 3 * 128 * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw5, 3 * 3 * 256 * 3   * sizeof(float)));

    // -------- BIAS GRADIENTS --------
    CUDA_CHECK(cudaMalloc(&db1, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db3, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db4, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db5, 3   * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    std::cout << "GPU memory allocated successfully\n";
    initWeights();
    std::cout << "Weights initialized successfully\n";
}

gpu_AE::~gpu_AE() {
    cudaFree(w1); cudaFree(w2); cudaFree(w3);
    cudaFree(w4); cudaFree(w5);
    cudaFree(b1); cudaFree(b2); cudaFree(b3);
    cudaFree(b4); cudaFree(b5);
    cudaFree(d_input);
    cudaFree(o1); cudaFree(o2); cudaFree(o3);
    cudaFree(o4); cudaFree(o5); cudaFree(o6);
    cudaFree(o7); cudaFree(o8); cudaFree(o9);
    cudaFree(d_o1); cudaFree(d_o2); cudaFree(d_o3);
    cudaFree(d_o4); cudaFree(d_o5); cudaFree(d_o6);
    cudaFree(d_o7); cudaFree(d_o8); cudaFree(d_o9);
    cudaFree(dw1); cudaFree(dw2); cudaFree(dw3);
    cudaFree(dw4); cudaFree(dw5);
    cudaFree(db1); cudaFree(db2); cudaFree(db3);
    cudaFree(db4); cudaFree(db5);
    cudaFree(d_loss);
}

void gpu_AE::initWeights() {
    std::mt19937 gen(42);

    auto he_init = [&](int fan_in, int size) {
        float stddev = sqrtf(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, stddev);
        std::vector<float> w(size);
        for (int i = 0; i < size; i++)
            w[i] = dist(gen);
        return w;
    };

    auto h_w1 = he_init(3 * 3 * 3, 3 * 3 * 3 * 256);
    CUDA_CHECK(cudaMemcpy(w1, h_w1.data(), h_w1.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto h_w2 = he_init(3 * 3 * 256, 3 * 3 * 256 * 128);
    CUDA_CHECK(cudaMemcpy(w2, h_w2.data(), h_w2.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto h_w3 = he_init(3 * 3 * 128, 3 * 3 * 128 * 128);
    CUDA_CHECK(cudaMemcpy(w3, h_w3.data(), h_w3.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto h_w4 = he_init(3 * 3 * 128, 3 * 3 * 128 * 256);
    CUDA_CHECK(cudaMemcpy(w4, h_w4.data(), h_w4.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto h_w5 = he_init(3 * 3 * 256, 3 * 3 * 256 * 3);
    CUDA_CHECK(cudaMemcpy(w5, h_w5.data(), h_w5.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(b1, 0, 256 * sizeof(float)));
    CUDA_CHECK(cudaMemset(b2, 0, 128 * sizeof(float)));
    CUDA_CHECK(cudaMemset(b3, 0, 128 * sizeof(float)));
    CUDA_CHECK(cudaMemset(b4, 0, 256 * sizeof(float)));
    CUDA_CHECK(cudaMemset(b5, 0, 3   * sizeof(float)));
}

void gpu_AE::forward(float* h_input) {
    int N = batch_size * 32 * 32 * 3;
    
    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input,
        N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block2d(16, 16);

    // ENCODER
    dim3 grid32_256((32+15)/16, (32+15)/16, batch_size*256);
    conv_naive<<<grid32_256, block2d>>>(d_input, w1, b1, o1, 32, 32, 3, 256);
    relu_naive<<<(batch_size*32*32*256 + 255)/256, 256>>>(o1, batch_size*32*32*256);

    dim3 grid16_256((16+15)/16, (16+15)/16, batch_size*256);
    maxpool2x2<<<grid16_256, block2d>>>(o1, o2, 32, 32, 256);

    dim3 grid16_128((16+15)/16, (16+15)/16, batch_size*128);
    conv_naive<<<grid16_128, block2d>>>(o2, w2, b2, o3, 16, 16, 256, 128);
    relu_naive<<<(batch_size*16*16*128 + 255)/256, 256>>>(o3, batch_size*16*16*128);

    dim3 grid8_128((8+15)/16, (8+15)/16, batch_size*128);
    maxpool2x2<<<grid8_128, block2d>>>(o3, o4, 16, 16, 128);

    // DECODER
    conv_naive<<<grid8_128, block2d>>>(o4, w3, b3, o5, 8, 8, 128, 128);
    relu_naive<<<(batch_size*8*8*128 + 255)/256, 256>>>(o5, batch_size*8*8*128);

    upsample2x_naive<<<grid16_128, block2d>>>(o5, o6, 8, 8, 128);

    dim3 grid16_256_v2((16+15)/16, (16+15)/16, batch_size*256);
    conv_naive<<<grid16_256_v2, block2d>>>(o6, w4, b4, o7, 16, 16, 128, 256);
    relu_naive<<<(batch_size*16*16*256 + 255)/256, 256>>>(o7, batch_size*16*16*256);

    dim3 grid32_256_v2((32+15)/16, (32+15)/16, batch_size*256);
    upsample2x_naive<<<grid32_256_v2, block2d>>>(o7, o8, 16, 16, 256);

    dim3 grid32_3((32+15)/16, (32+15)/16, batch_size*3);
    conv_naive<<<grid32_3, block2d>>>(o8, w5, b5, o9, 32, 32, 256, 3);

    // Compute loss
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    mse_loss_kernel<<<(N + 255)/256, 256>>>(o9, d_input, d_loss, N);

    // ONLY sync once at the end
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_AE::backward() {
    const int N  = batch_size * 32 * 32 * 3;
    const int BS = batch_size;

    // ✅ ONLY CLEAR WEIGHT/BIAS GRADIENTS (NOT activation gradients)
    cudaMemset(dw1, 0, 3*3*3*256*sizeof(float));
    cudaMemset(dw2, 0, 3*3*256*128*sizeof(float));
    cudaMemset(dw3, 0, 3*3*128*128*sizeof(float));
    cudaMemset(dw4, 0, 3*3*128*256*sizeof(float));
    cudaMemset(dw5, 0, 3*3*256*3*sizeof(float));
    cudaMemset(db1, 0, 256*sizeof(float));
    cudaMemset(db2, 0, 128*sizeof(float));
    cudaMemset(db3, 0, 128*sizeof(float));
    cudaMemset(db4, 0, 256*sizeof(float));
    cudaMemset(db5, 0, 3*sizeof(float));

    // MSE gradient
    mse_backward<<<(N+255)/256, 256>>>(o9, d_input, d_o9, N);

    // Conv5 backward
    conv_weight_backward<<<dim3(3,256), dim3(3,3)>>>(
        o8, d_o9, dw5, 32, 32, 256, 3, BS);
    conv_bias_backward<<<3, 256>>>(
        d_o9, db5, 32, 32, 3, BS);
    conv_input_backward<<<dim3((32+15)/16,(32+15)/16,BS*256), dim3(16,16)>>>(
        d_o9, w5, d_o8, 32, 32, 256, 3);

    // Upsample2 backward
    dim3 blk(16,16);
    upsample_backward<<<dim3((16+15)/16,(16+15)/16,BS*256), blk>>>(
        d_o8, d_o7, 16, 16, 256);

    // ReLU4 + Conv4 backward
    relu_backward<<<(BS*16*16*256+255)/256,256>>>(o7, d_o7, d_o7, BS*16*16*256);
    conv_weight_backward<<<dim3(256,128), dim3(3,3)>>>(
        o6, d_o7, dw4, 16, 16, 128, 256, BS);
    conv_bias_backward<<<256, 256>>>(
        d_o7, db4, 16, 16, 256, BS);
    conv_input_backward<<<dim3((16+15)/16,(16+15)/16,BS*128), dim3(16,16)>>>(
        d_o7, w4, d_o6, 16, 16, 128, 256);

    // Upsample1 backward
    upsample_backward<<<dim3((8+15)/16,(8+15)/16,BS*128), blk>>>(
        d_o6, d_o5, 8, 8, 128);

    // ReLU3 + Conv3 backward
    relu_backward<<<(BS*8*8*128+255)/256,256>>>(o5, d_o5, d_o5, BS*8*8*128);
    conv_weight_backward<<<dim3(128,128), dim3(3,3)>>>(
        o4, d_o5, dw3, 8, 8, 128, 128, BS);
    conv_bias_backward<<<128, 256>>>(
        d_o5, db3, 8, 8, 128, BS);
    conv_input_backward<<<dim3((8+15)/16,(8+15)/16,BS*128), dim3(16,16)>>>(
        d_o5, w3, d_o4, 8, 8, 128, 128);

    // Pool2 backward
    maxpool_backward<<<dim3((16+15)/16,(16+15)/16,BS*128), blk>>>(
        o3, d_o4, d_o3, 16, 16, 128);

    // ReLU2 + Conv2 backward
    relu_backward<<<(BS*16*16*128+255)/256,256>>>(o3, d_o3, d_o3, BS*16*16*128);
    conv_weight_backward<<<dim3(128,256), dim3(3,3)>>>(
        o2, d_o3, dw2, 16, 16, 256, 128, BS);
    conv_bias_backward<<<128, 256>>>(
        d_o3, db2, 16, 16, 128, BS);
    conv_input_backward<<<dim3((16+15)/16,(16+15)/16,BS*256), dim3(16,16)>>>(
        d_o3, w2, d_o2, 16, 16, 256, 128);

    // Pool1 backward
    maxpool_backward<<<dim3((16+15)/16,(16+15)/16,BS*256), blk>>>(
        o1, d_o2, d_o1, 32, 32, 256);

    // ReLU1 + Conv1 backward
    relu_backward<<<(BS*32*32*256+255)/256,256>>>(o1, d_o1, d_o1, BS*32*32*256);
    conv_weight_backward<<<dim3(256,3), dim3(3,3)>>>(
        d_input, d_o1, dw1, 32, 32, 3, 256, BS);
    conv_bias_backward<<<256, 256>>>(
        d_o1, db1, 32, 32, 256, BS);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_AE::train(CIFAR10Dataset& dataset, int epochs, float lr) {
    const int num_batches = dataset.getNumTrainBatches(batch_size);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\n========== Epoch " << epoch + 1 << "/" << epochs << " ==========\n";
        dataset.shuffle();

        float epoch_loss = 0.0f;
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < num_batches; i++) {
            float* batch = dataset.getBatch(i * batch_size, batch_size, true);

            forward(batch);
            backward();

            // SGD update ALL layers
            sgd_update<<<(3*3*3*256 + 255)/256, 256>>>(w1, dw1, lr, 3*3*3*256);
            sgd_update<<<(256 + 255)/256, 256>>>(b1, db1, lr, 256);
            
            sgd_update<<<(3*3*256*128 + 255)/256, 256>>>(w2, dw2, lr, 3*3*256*128);
            sgd_update<<<(128 + 255)/256, 256>>>(b2, db2, lr, 128);
            
            sgd_update<<<(3*3*128*128 + 255)/256, 256>>>(w3, dw3, lr, 3*3*128*128);
            sgd_update<<<(128 + 255)/256, 256>>>(b3, db3, lr, 128);
            
            sgd_update<<<(3*3*128*256 + 255)/256, 256>>>(w4, dw4, lr, 3*3*128*256);
            sgd_update<<<(256 + 255)/256, 256>>>(b4, db4, lr, 256);
            
            sgd_update<<<(3*3*256*3 + 255)/256, 256>>>(w5, dw5, lr, 3*3*256*3);
            sgd_update<<<(3 + 255)/256, 256>>>(b5, db5, lr, 3);

            delete[] batch;

            // ✅ Only copy loss every 100 batches
            if (i % 100 == 100 || i == num_batches - 1) {
                float batch_loss;
                CUDA_CHECK(cudaMemcpy(&batch_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
                batch_loss /= (batch_size * 32 * 32 * 3);
                epoch_loss += batch_loss * 100; // Approximate
                std::cout << "  Batch " << i << "/" << num_batches 
                          << " | Loss = " << batch_loss << std::endl;
            }
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        std::cout << "Epoch " << epoch + 1 
                  << " completed in " << ms/1000.0f << " seconds" << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
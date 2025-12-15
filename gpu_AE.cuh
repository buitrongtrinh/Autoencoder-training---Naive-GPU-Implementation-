#pragma once
#include <cuda_runtime.h>
#include "dataset.cuh"

class gpu_AE {
private:
    int batch_size;

    /* ===== INPUT ===== */
    float* d_input;

    /* ===== FORWARD OUTPUTS ===== */
    float *o1, *o2, *o3, *o4, *o5, *o6, *o7, *o8, *o9;

    /* ===== WEIGHTS ===== */
    float *w1, *w2, *w3, *w4, *w5;
    float *b1, *b2, *b3, *b4, *b5;

    /* ===== LOSS ===== */
    float* d_loss;

    /* ===== BACKWARD GRADIENTS ===== */
    float *d_o1, *d_o2, *d_o3, *d_o4, *d_o5;
    float *d_o6, *d_o7, *d_o8, *d_o9;

    float *dw1, *dw2, *dw3, *dw4, *dw5;
    float *db1, *db2, *db3, *db4, *db5;

public:
    gpu_AE(int batch_size);
    ~gpu_AE();

    void initWeights();
    void forward(float* h_input);
    void backward();
    void train(CIFAR10Dataset& dataset, int epochs, float lr);
    float* getDeviceLoss() { return d_loss; }
};

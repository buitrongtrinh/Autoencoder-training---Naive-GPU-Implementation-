#include "gpu_AE.cuh"
#include "dataset.cuh"
#include <iostream>

int main() {

    const int BATCH_SIZE = 64;
    const int EPOCHS = 5;
    const float LR = 1e-4f;
    const char* DATA_PATH = "cifar-10-batches-bin";

    std::cout << "=== LOAD DATASET ===" << std::endl;

    CIFAR10Dataset dataset;
    if (!dataset.loadData(DATA_PATH)) {
        std::cerr << "Load CIFAR-10 failed!" << std::endl;
        return -1;
    }

    std::cout << "=== INIT GPU AUTOENCODER ===" << std::endl;
    gpu_AE ae(BATCH_SIZE);

    std::cout << "=== START TRAINING ===" << std::endl;
    ae.train(dataset, EPOCHS, LR);

    std::cout << "=== DONE ===" << std::endl;
    return 0;
}

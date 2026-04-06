#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// nvcc -arch=sm_75 

__global__ void matmul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    const int N = 512;
    const size_t bytes = N * N * sizeof(float);

    // Host data
    std::vector<float> host_A(N * N);
    std::vector<float> host_B(N * N);
    std::vector<float> host_C(N * N, 0.0f);

    for (int i = 0; i < N * N; ++i) {
        host_A[i] = static_cast<float>(rand()) / RAND_MAX;
        host_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory allocation
    float *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, bytes);
    cudaMalloc(&dev_B, bytes);
    cudaMalloc(&dev_C, bytes);

    // Copy to device
    cudaMemcpy(dev_A, host_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B.data(), bytes, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul<<<grid_size, block_size>>>(dev_A, dev_B, dev_C, N);
    cudaEventRecord(stop);
    
    // Copy result back
    cudaMemcpy(host_C.data(), dev_C, bytes, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    std::cout << "Matrix size: " << N << " x " << N << std::endl;
    std::cout << "Execution time: " << elapsed_ms / 1000.0 << " seconds" << std::endl;

    // Cleanup
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
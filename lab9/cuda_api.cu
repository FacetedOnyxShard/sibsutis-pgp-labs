#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// Ядро для перемножения матриц
__global__ void matmul(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 512;
    size_t size = N * N * sizeof(float);

    // 1. Выделение памяти на хосте (CPU) и инициализация
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N, 0.0f);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 2. Выделение памяти на девайсе (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. Копирование данных с CPU на GPU
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // 4. Настройка сетки и блоков
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Настройка событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- НАЧАЛО ЗАМЕРА ВРЕМЕНИ ---
    cudaEventRecord(start);
    
    // 5. Запуск ядра
    matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop);
    // --- КОНЕЦ ЗАМЕРА ВРЕМЕНИ ---

    // 6. Копирование результата обратно на CPU
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Синхронизация и расчет времени
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Размер матриц: " << N << " x " << N << std::endl;
    std::cout << "Время выполнения (CUDA Runtime API): " << milliseconds / 1000.0 << " секунд" << std::endl;

    // 7. Очистка памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
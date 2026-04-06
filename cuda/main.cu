#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

constexpr int N 1024
constexpr int BLOCK_SIZE 16

__global__ void matmul_kernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    size_t matrix_size = N * N;
    size_t mat_size_in_mem = matrix_size * sizeof(float);
    float *h_A = (float*)malloc(mat_size_in_mem);
    float *h_B = (float*)malloc(mat_size_in_mem);
    float *h_C = (float*)malloc(mat_size_in_mem);

    // Инициализация матриц
    for (int i = 0; i < matrix_size; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, mat_size_in_mem);
    cudaMalloc(&d_B, mat_size_in_mem);
    cudaMalloc(&d_C, mat_size_in_mem);

    cudaMemcpy(d_A, h_A, mat_size_in_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mat_size_in_mem, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Runtime API time: %f ms\n", elapsed_ms);

    cudaMemcpy(h_C, d_C, mat_size_in_mem, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
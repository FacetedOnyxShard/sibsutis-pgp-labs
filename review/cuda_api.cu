#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

using namespace std;

constexpr int STD_BLOCK_SIDE_SIZE_FOR_MATRIX = 16;

// nvcc -arch=sm_75 cuda_api.cu && ./a.out

__global__ void matmul(const float *a, 
  const float *b, float *c, const size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float value = 0.f;
    for (size_t k = 0; k < N; ++k) {
      value += a[row * N + k] * b[k * N + col];
    }
    c[row * N + col] = value;
}

int main() {
  srand(time(NULL));

  constexpr size_t N = 1024;
  constexpr size_t msize = N * N;
  constexpr size_t memsize = N * N * sizeof(float);

  vector<float> host_a(msize);
  vector<float> host_b(msize);
  vector<float> host_c(msize, 0.f);
  vector<float> cpu_c(msize, 0.f);

  for (size_t i = 0; i < msize; ++i) {
    host_a[i] = static_cast<float>(rand()) / RAND_MAX;
    host_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *device_a;
  float *device_b;
  float *device_c;
  cudaMalloc(&device_a, memsize);
  cudaMalloc(&device_b, memsize);
  cudaMalloc(&device_c, memsize);

  cudaMemcpy(device_a, host_a.data(), memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b.data(), memsize, cudaMemcpyHostToDevice);

  dim3 block_size(STD_BLOCK_SIDE_SIZE_FOR_MATRIX, STD_BLOCK_SIDE_SIZE_FOR_MATRIX);
  dim3 grid_size((N + block_size.x - 1) / block_size.x,
                (N + block_size.y - 1) / block_size.y);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matmul<<<grid_size, block_size>>>(device_a, device_b, device_c, N);
  cudaEventRecord(stop);

  cudaMemcpy(host_c.data(), device_c, memsize, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float elapsed_ms = 0;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  cout << "Matrix size: " << N << "x" << N << '\n';
  cout << "Execution time: " << elapsed_ms / 1000.0 << " seconds\n";

  // cleanup
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
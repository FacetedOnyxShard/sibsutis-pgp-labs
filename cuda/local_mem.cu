#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

constexpr int N = 4096;
constexpr int K = 4096;
constexpr int MATRIX_SIZE = N * K;

__managed__ float a[N * K];
__managed__ float b[K * N];

__global__ void kernel() {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  constexpr int c_size = 1024;
  volatile float c[c_size];
  // просто чтобы компилятор не оптимизировал
  for (int i = 0; i < c_size; ++i) {
    c[i] = i * 0.001f;
  }

  if (idx < MATRIX_SIZE) {
    int row = idx / K;
    int col = idx % K;

    b[col * N + row] = a[row * K + col] + c[idx % c_size];
  }
}

int main() {
  srand(time(NULL));

  constexpr int threadsInBlock = 256;
  constexpr int blockCount = (MATRIX_SIZE + (threadsInBlock - 1)) / threadsInBlock;
  kernel<<<blockCount, threadsInBlock>>>();
  cudaDeviceSynchronize();

  return 0;
}
#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

constexpr int N = 3;
constexpr int K = 4;
constexpr int MATRIX_SIZE = N * K;

__managed__ float a[N * K];
__managed__ float b[K * N];

__global__ void kernel() {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < MATRIX_SIZE) {
    int row = idx / K;
    int col = idx % K;

    b[col * N + row] = a[row * K + col];
  }
}

int main() {
  srand(time(NULL));

  for (size_t i = 0; i < N * K; ++i) {
    a[i] = i + 1;
    printf("%0.2f ", a[i]);
    if ((i + 1) % K == 0) printf("\n");
  }
  printf("\n\n");

  constexpr int threadsInBlock = 256;
  constexpr int blockCount = (MATRIX_SIZE + (threadsInBlock - 1)) / threadsInBlock;
  kernel<<<blockCount, threadsInBlock>>>();
  cudaDeviceSynchronize();


  for (size_t i = 0; i < N * K; ++i) {
    printf("%0.2f ", b[i]);
    if ((i + 1) % N == 0) printf("\n");
  }
  printf("\n\n");

  return 0;
}
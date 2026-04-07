// nvcc --ptx cuda_driver_kernels.cu -o kernels.ptx

extern "C" __global__ void matmul(const float *a, 
  const float *b, float *c, size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float value = 0.f;
    for (size_t k = 0; k < N; ++k) {
      value += a[row * N + k] * b[k * N + col];
    }
    c[row * N + col] = value;
}
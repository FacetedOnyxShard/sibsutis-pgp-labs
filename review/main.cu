#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

constexpr int N = (1 << 20) * 10; // 10М стандарт для измерений
constexpr int BLOCK_SIZE = 256;
constexpr int ITER = 5;

__global__ void vadd(const float *a, const float *b, float *c, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    c[i] = a[i] + b[i];
}

__global__ void vdot(const float *a, const float *b, float *c, int N) {
  __shared__ float cache[BLOCK_SIZE];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = threadIdx.x;
  float sum = 0;

  while (tid < N) {
    sum += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  cache[i] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (i < s)
      cache[i] += cache[i + s];
    __syncthreads();
  }

  if (i == 0)
    atomicAdd(c, cache[0]);
}

void copy_test(size_t bytes, bool use_pinned = false) {
  float *host_data, *device_data;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (use_pinned) {
    cudaHostAlloc(&host_data, bytes, cudaHostAllocDefault);
  } else {
    host_data = (float *)malloc(bytes);
  }
  cudaMalloc(&device_data, bytes);

  // host to device
  cudaEventRecord(start);
  cudaMemcpy(device_data, host_data, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float h2d_ms;
  cudaEventElapsedTime(&h2d_ms, start, stop);

  // device to host
  cudaEventRecord(start);
  cudaMemcpy(host_data, device_data, bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float d2h_ms;
  cudaEventElapsedTime(&d2h_ms, start, stop);

  cout << left << setw(20) << (use_pinned ? "Pinned" : "Regular") << setw(20)
       << h2d_ms << setw(20) << d2h_ms << endl;

  // cleanup
  if (use_pinned)
    cudaFreeHost(host_data);
  else
    free(host_data);
  cudaFree(device_data);
}

void bench(size_t bytes, const string &name,
           bool is_dot = false) {
  float *ha, *hb, *hc = nullptr, *da, *db, *dc = nullptr;

  cudaHostAlloc(&ha, bytes, cudaHostAllocDefault);
  cudaHostAlloc(&hb, bytes, cudaHostAllocDefault);

  for (int i = 0; i < N; i++) {
    ha[i] = rand() / (float)RAND_MAX;
    hb[i] = rand() / (float)RAND_MAX;
  }

  if (!is_dot) {
    cudaHostAlloc(&hc, bytes, cudaHostAllocDefault);
    cudaMalloc(&dc, bytes);
  }

  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);

  float *h_sums = nullptr, *d_sums = nullptr;
  if (is_dot) {
    cudaHostAlloc(&h_sums, 16 * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_sums, 16 * sizeof(float));
  }

  cout << "\n" << name << "\n";
  cout << setw(20) << "Streams" << setw(20) << "Chunk" << "Time (ms)\n";

  for (int s : {1, 2, 4, 8, 16}) {
    vector<cudaStream_t> streams(s);
    for (int i = 0; i < s; ++i)
      cudaStreamCreate(&streams[i]);

    int chunk = N / s;
    size_t chunk_bytes = chunk * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;

    for (int it = 0; it < ITER; ++it) {
      cudaEventRecord(start);

      for (int i = 0; i < s; ++i) {
        int offset = i * chunk;

        cudaMemcpyAsync(&da[offset], &ha[offset], chunk_bytes,
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&db[offset], &hb[offset], chunk_bytes,
                        cudaMemcpyHostToDevice, streams[i]);

        int grid = min((chunk + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);

        if (is_dot) {
          cudaMemsetAsync(&d_sums[i], 0, sizeof(float), streams[i]);
          vdot<<<grid, BLOCK_SIZE, 0, streams[i]>>>(&da[offset], &db[offset],
                                                      &d_sums[i], chunk);
          cudaMemcpyAsync(&h_sums[i], &d_sums[i], sizeof(float),
                          cudaMemcpyDeviceToHost, streams[i]);
        } else {
          vadd<<<grid, BLOCK_SIZE, 0, streams[i]>>>(&da[offset], &db[offset],
                                                      &dc[offset], chunk);
          cudaMemcpyAsync(&hc[offset], &dc[offset], chunk_bytes,
                          cudaMemcpyDeviceToHost, streams[i]);
        }
      }
      cudaDeviceSynchronize();

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms;
      cudaEventElapsedTime(&ms, start, stop);
      total += ms;
    }

    cout << setw(20) << s << setw(20) << chunk << total / ITER << endl;

    for (auto &st : streams)
      cudaStreamDestroy(st);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // cleanup
  cudaFreeHost(ha);
  cudaFreeHost(hb);
  if (!is_dot) {
    cudaFreeHost(hc);
    cudaFree(dc);
  }
  cudaFree(da);
  cudaFree(db);
  if (is_dot) {
    cudaFreeHost(h_sums);
    cudaFree(d_sums);
  }
}

// nvcc -arch=sm_75 main.cu && ./a.out

int main() {
  size_t bytes = N * sizeof(float);
  
  cout << "Copy performance (ms)\n";
  cout << left << setw(20) << "Type" << setw(20) << "H->D" << setw(20)
       << "D->H" << endl;

  copy_test(bytes);
  copy_test(bytes, true);

  bench(bytes, "Vector addition");
  bench(bytes, "Dot product", true);

  return 0;
}
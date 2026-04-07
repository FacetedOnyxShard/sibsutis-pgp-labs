#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>

constexpr int STD_BLOCK_SIDE_SIZE_FOR_MATRIX = 16;

using namespace std;

// nvcc cuda_driver.cpp -lcuda

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

  cuInit(0); // 0 флаг
  CUdevice device;
  cuDeviceGet(&device, 0); // 0 индекс устройства

  CUcontext ctx;
  cuDevicePrimaryCtxRetain(&ctx, device);
  cuCtxSetCurrent(ctx);

  CUmodule module;
  CUresult res = cuModuleLoad(&module, "kernels.ptx");
  if (res != CUDA_SUCCESS) {
    cerr << "Не удалось загрузить ptx файл\n";
    return -1;
  }

  CUfunction kernel;
  cuModuleGetFunction(&kernel, module, "matmul");

  CUdeviceptr device_a, device_b, device_c;
  cuMemAlloc(&device_a, memsize);
  cuMemAlloc(&device_b, memsize);
  cuMemAlloc(&device_c, memsize);

  cuMemcpyHtoD(device_a, host_a.data(), memsize);
  cuMemcpyHtoD(device_b, host_b.data(), memsize);

  constexpr int block_size = STD_BLOCK_SIDE_SIZE_FOR_MATRIX;
  constexpr int grid_size = (N + block_size - 1) / block_size;

  CUevent start, stop;
  cuEventCreate(&start, CU_EVENT_DEFAULT);
  cuEventCreate(&stop, CU_EVENT_DEFAULT);

  size_t N_for_args = N;
  void *args[] = {&device_a, &device_b, &device_c, &N_for_args};

  cuEventRecord(start, NULL);               // null - поток по умолчанию
  cuLaunchKernel(kernel,                    //
                 grid_size, grid_size, 1,   //
                 block_size, block_size, 1, //
                 0, NULL,                   //
                 args,
                 NULL); // x,y,z, размер sharedMem, поток для async,args, extra
  cuEventRecord(stop, NULL);
  cuEventSynchronize(stop);

  float elapsed_ms = 0;
  cuEventElapsedTime(&elapsed_ms, start, stop);
  cuMemcpyDtoH(host_c.data(), device_c, memsize);

  cout << "Matrix size: " << N << "x" << N << '\n';
  cout << "Execution time (cuda driver api): " << elapsed_ms / 1000.0
       << " seconds\n";

  // cleanup
  cuMemFree(device_a);
  cuMemFree(device_b);
  cuMemFree(device_c);
  cuEventDestroy(start);
  cuEventDestroy(stop);
  cuCtxDestroy(ctx);
  return 0;
}
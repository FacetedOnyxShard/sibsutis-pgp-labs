import sys
from ctypes import *
import numpy as np
import time
import os

cuda = CDLL('libcuda.so')
CUDA_ERRORS = {0 : 'CUDA_SUCCESS', 1 : 'CUDA_ERROR_INVALID_VALUE', 200 : 'CUDA_ERROR_INVALID_IMAGE', 201 : 'CUDA_ERROR_INVALID_CONTEXT ', 400 : 'CUDA_ERROR_INVALID_HANDLE' }

cuInit = cuda.cuInit
cuInit.argtypes = [c_uint]
cuInit.restype = int

cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]
cuDeviceGetCount.restype = int

cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet.restype = int

cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [POINTER(c_void_p), c_uint, c_int]
cuCtxCreate.restype = int

cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [c_void_p, c_char_p]
cuModuleLoad.restype  = int

cuCtxSynchronize = cuda.cuCtxSynchronize
cuCtxSynchronize.argtypes = []
cuCtxSynchronize.restype = int

cuModuleGetFunction = cuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p ]
cuModuleGetFunction.restype = int

cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyHtoD = cuda.cuMemcpyHtoD 
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemAlloc.restype = int

cuMemcpyDtoH = cuda.cuMemcpyDtoH 
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyDtoH.restype = int

cuMemFree = cuda.cuMemFree
cuMemFree.argtypes = [c_void_p] 
cuMemFree.restype = int

cuLaunchKernel = cuda.cuLaunchKernel
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p]
cuLaunchKernel.restype = int

cuCtxDestroy = cuda.cuCtxDestroy
cuCtxDestroy.argtypes = [c_void_p]
cuCtxDestroy.restype = int

# Создание события
cuEventCreate = cuda.cuEventCreate
cuEventCreate.argtypes = [POINTER(c_void_p), c_uint]
cuEventCreate.restype = int

# Запись события в поток
cuEventRecord = cuda.cuEventRecord
cuEventRecord.argtypes = [c_void_p, c_void_p]  # CUevent, CUstream
cuEventRecord.restype = int

# Ожидание завершения события
cuEventSynchronize = cuda.cuEventSynchronize
cuEventSynchronize.argtypes = [c_void_p]
cuEventSynchronize.restype = int

# Вычисление времени между событиями (в миллисекундах)
cuEventElapsedTime = cuda.cuEventElapsedTime
cuEventElapsedTime.argtypes = [POINTER(c_float), c_void_p, c_void_p]
cuEventElapsedTime.restype = int

# Уничтожение события
cuEventDestroy = cuda.cuEventDestroy
cuEventDestroy.argtypes = [c_void_p]
cuEventDestroy.restype = int

# Константы для флагов событий
CU_EVENT_DEFAULT = 0
CU_EVENT_BLOCKING_SYNC = 1
CU_EVENT_DISABLE_TIMING = 2

STD_BLOCK_SIDE_SIZE_FOR_MATRIX = 16

def main():
  np.random.seed(int(time.time()))
  ptx_file = b"kernels.ptx"

  N = 1024
  msize = N * N
  memsize = N * N * sizeof(c_float)

  host_a = np.random.rand(N, N).astype(np.float32)
  host_b = np.random.rand(N, N).astype(np.float32)
  host_c = np.zeros((N, N), dtype=np.float32)

  cuInit(0)

  device = c_int()
  cuDeviceGet(byref(device), 0)

  ctx = c_void_p()
  cuCtxCreate(byref(ctx), 0, device)

  module = c_void_p()
  cuModuleLoad(byref(module), ptx_file)

  kernel = c_void_p()
  cuModuleGetFunction(byref(kernel), module, b"matmul")

  device_a = c_void_p()
  device_b = c_void_p()
  device_c = c_void_p()
  
  cuMemAlloc(byref(device_a), memsize)
  cuMemAlloc(byref(device_b), memsize)
  cuMemAlloc(byref(device_c), memsize)

  cuMemcpyHtoD(device_a, host_a.ctypes.data_as(c_void_p), memsize)
  cuMemcpyHtoD(device_b, host_b.ctypes.data_as(c_void_p), memsize)

  c_var_N = c_int(N)
  args = (c_void_p * 4)(
    cast(byref(device_a), c_void_p),
    cast(byref(device_b), c_void_p),
    cast(byref(device_c), c_void_p),
    cast(byref(c_var_N), c_void_p)
  )

  start_event = c_void_p()
  end_event = c_void_p()

  cuEventCreate(byref(start_event), CU_EVENT_DEFAULT)
  cuEventCreate(byref(end_event), CU_EVENT_DEFAULT)

  block_size = STD_BLOCK_SIDE_SIZE_FOR_MATRIX
  grid_size = (N + block_size - 1) // block_size

  cuEventRecord(start_event, 0) # 0 = NULL stream, последовательное выполнение
  cuLaunchKernel(kernel, 
                grid_size, grid_size, 1,
                block_size, block_size, 1, 
                0, None, 
                args, None)
  cuEventRecord(end_event, 0)

  cuEventSynchronize(end_event)

  elapsed_ms = c_float()
  cuEventElapsedTime(byref(elapsed_ms), start_event, end_event)

  elapsed_sec = elapsed_ms.value / 1000.0

  cuMemcpyDtoH(host_c.ctypes.data_as(c_void_p), device_c, memsize)

  print(f"Размер матриц: {N}x{N}")
  print(f"Время выполнения (CUDA Events): {elapsed_sec:.5f} секунд")
  
  # cleanup
  cuEventDestroy(start_event)
  cuEventDestroy(end_event)
  cuMemFree(device_a)
  cuMemFree(device_b)
  cuMemFree(device_c)
  cuCtxDestroy(ctx)


if __name__ == "__main__":
  main()
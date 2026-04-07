import numpy as np
import time
from numba import cuda
import gc


@cuda.jit
def matmul(a, b, c, N):
  row, col = cuda.grid(2) # указываем размерность dim

  if row >= N or col >= N: return

  tmp = 0.0
  for i in range(N):
    tmp += a[row, i] * b[i, col]
  c[row, col] = tmp

STD_BLOCK_SIDE_SIZE_FOR_MATRIX = 16

def main():
  np.random.seed(int(time.time()))
  N = 1024

  host_a = np.random.rand(N, N).astype(np.float32)
  host_b = np.random.rand(N, N).astype(np.float32)
  host_c = np.zeros((N, N), dtype=np.float32)

  device_a = cuda.to_device(host_a)
  device_b = cuda.to_device(host_b)
  device_c = cuda.to_device(host_c)

  block_size = (STD_BLOCK_SIDE_SIZE_FOR_MATRIX, 
                STD_BLOCK_SIDE_SIZE_FOR_MATRIX)
  grid_size = ((N + block_size[0] - 1) // block_size[0], 
               (N + block_size[1] - 1) // block_size[1])
  
  start_event = cuda.event()
  end_event = cuda.event()

  start_event.record()
  matmul[grid_size, block_size](device_a, device_b, device_c, N)
  end_event.record()
  end_event.synchronize()

  elapsed_ms = cuda.event_elapsed_time(start_event, end_event)

  device_c.copy_to_host(host_c)

  print(f"Размер матриц: {N}x{N}")
  print(f"Время выполнения: {elapsed_ms / 1000.0} sec")

if __name__ == '__main__':
  main()
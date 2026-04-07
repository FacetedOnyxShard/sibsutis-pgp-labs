import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time

# !pip install pycuda

STD_BLOCK_SIDE_SIZE_FOR_MATRIX = 16

def main():
    np.random.seed(int(time.time()))
    
    N = 1024
    msize = N * N
    memsize = N * N * np.dtype(np.float32).itemsize
    
    host_a = np.random.rand(N, N).astype(np.float32)
    host_b = np.random.rand(N, N).astype(np.float32)
    host_c = np.zeros((N, N), dtype=np.float32)
    
    device_a = cuda.mem_alloc(host_a.nbytes)
    device_b = cuda.mem_alloc(host_b.nbytes)
    device_c = cuda.mem_alloc(host_c.nbytes)
    
    cuda.memcpy_htod(device_a, host_a)
    cuda.memcpy_htod(device_b, host_b)
    
    kernel_code = """
    __global__ void matmul(const float *a, const float *b, float *c, const size_t N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row >= N || col >= N) return;
        
        float value = 0.f;
        for (size_t k = 0; k < N; ++k) {
            value += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = value;
    }
    """
    
    mod = SourceModule(kernel_code)
    matmul_kernel = mod.get_function("matmul")
    
    block_size = (STD_BLOCK_SIDE_SIZE_FOR_MATRIX, STD_BLOCK_SIDE_SIZE_FOR_MATRIX, 1)
    grid_size = ((N + block_size[0] - 1) // block_size[0],
                 (N + block_size[1] - 1) // block_size[1], 1)
    
    start_event = cuda.Event()
    end_event = cuda.Event()
    
    start_event.record()
    matmul_kernel(
        device_a, device_b, device_c, np.int32(N),
        block=block_size,
        grid=grid_size
    )
    end_event.record()
    end_event.synchronize()
    
    
    elapsed_ms = start_event.time_till(end_event)
    
    
    cuda.memcpy_dtoh(host_c, device_c)
    
    print(f"Matrix size: {N}x{N}")
    print(f"Execution time (PyCUDA): {elapsed_ms / 1000.0:.5f} seconds")

if __name__ == "__main__":
    main()
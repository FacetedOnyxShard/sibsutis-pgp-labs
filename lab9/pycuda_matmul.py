import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time

def main():
    N = 512

    # 1. Подготовка данных на хосте (CPU)
    h_A = np.random.rand(N, N).astype(np.float32)
    h_B = np.random.rand(N, N).astype(np.float32)
    h_C = np.zeros((N, N), dtype=np.float32)

    # 2. Выделение памяти на видеокарте и копирование данных
    # PyCuda делает это гораздо лаконичнее
    d_A = cuda.mem_alloc(h_A.nbytes)
    d_B = cuda.mem_alloc(h_B.nbytes)
    d_C = cuda.mem_alloc(h_C.nbytes)

    cuda.memcpy_htod(d_A, h_A)
    cuda.memcpy_htod(d_B, h_B)

    # 3. Написание и компиляция ядра "на лету"
    mod = SourceModule("""
    __global__ void matmul(const float *A, const float *B, float *C, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < N && col < N) {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum += A[row * N + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """)

    # Получаем скомпилированную функцию
    matmul = mod.get_function("matmul")

    # 4. Настройка сетки и блоков
    block_size = 16
    grid_size = (N + block_size - 1) // block_size

    # --- НАЧАЛО ЗАМЕРА ВРЕМЕНИ ---
    start_time = time.time()

    # 5. Запуск ядра
    matmul(
        d_A, d_B, d_C, np.int32(N), 
        block=(block_size, block_size, 1), 
        grid=(grid_size, grid_size, 1)
    )
    
    # Синхронизация для точного замера времени
    cuda.Context.synchronize()

    # --- КОНЕЦ ЗАМЕРА ВРЕМЕНИ ---
    end_time = time.time()

    # 6. Копирование результата обратно на CPU
    cuda.memcpy_dtoh(h_C, d_C)

    print(f"Размер матриц: {N} x {N}")
    print(f"Время выполнения (PyCuda): {end_time - start_time:.5f} секунд")

if __name__ == '__main__':
    main()
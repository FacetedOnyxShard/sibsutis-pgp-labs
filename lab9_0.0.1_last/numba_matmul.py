import numpy as np
import time
from numba import cuda

# 1. Пишем ядро прямо на Python! Декоратор сделает всю магию компиляции
@cuda.jit
def matmul_numba(A, B, C, N):
    # Удобное получение глобальных индексов потока
    row, col = cuda.grid(2)
    
    if row < N and col < N:
        tmp = 0.0
        for i in range(N):
            tmp += A[row, i] * B[i, col]
        C[row, col] = tmp

def main():
    N = 512

    # Подготовка данных на хосте (CPU)
    h_A = np.random.rand(N, N).astype(np.float32)
    h_B = np.random.rand(N, N).astype(np.float32)
    h_C = np.zeros((N, N), dtype=np.float32)

    # 2. Копирование данных на видеокарту (Device)
    # Numba делает это буквально в одну строчку
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    d_C = cuda.to_device(h_C)

    # 3. Настройка сетки и блоков
    block_size = (16, 16)
    grid_size_x = (N + block_size[0] - 1) // block_size[0]
    grid_size_y = (N + block_size[1] - 1) // block_size[1]
    grid_size = (grid_size_x, grid_size_y)

    # --- НАЧАЛО ЗАМЕРА ВРЕМЕНИ ---
    # Numba компилирует ядро при первом вызове, поэтому первый запуск
    # всегда включает время компиляции. Чтобы замер был честным, 
    # в реальных задачах делают "прогревочный" вызов, но оставим как есть для сравнения.
    start_time = time.time()

    # 4. Запуск ядра
    matmul_numba[grid_size, block_size](d_A, d_B, d_C, N)
    
    # Ожидание завершения работы GPU
    cuda.synchronize()

    # --- КОНЕЦ ЗАМЕРА ВРЕМЕНИ ---
    end_time = time.time()

    # 5. Копирование результата обратно на CPU
    d_C.copy_to_host(h_C)

    print(f"Размер матриц: {N} x {N}")
    print(f"Время выполнения (Numba): {end_time - start_time:.5f} секунд")

if __name__ == '__main__':
    main()
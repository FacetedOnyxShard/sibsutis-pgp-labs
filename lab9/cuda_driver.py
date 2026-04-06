import sys
from ctypes import *
import numpy as np
import time

# === ВАШ КОД ИЗ ЛЕКЦИИ ===
if 'linux' in sys.platform:
    cuda = CDLL('libcuda.so')
elif 'win' in sys.platform:
    cuda = CDLL('nvcuda.dll')

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
cuCtxCreate.argtypes = [c_void_p, c_uint, c_int]
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


# === ОСНОВНАЯ ФУНКЦИЯ ДЛЯ ВЫПОЛНЕНИЯ ЗАДАНИЯ ===
def main():
    # Размерность матриц (N x N)
    N = 512
    # Теперь sizeof и c_float берутся из "from ctypes import *" в начале файла
    size = N * N * sizeof(c_float)

    # 1. Подготавливаем матрицы на хосте (CPU)
    h_A = np.random.rand(N, N).astype(np.float32)
    h_B = np.random.rand(N, N).astype(np.float32)
    h_C = np.zeros((N, N), dtype=np.float32)

    # 2. Инициализация CUDA
    cuInit(0)
    
    device = c_int()
    cuDeviceGet(byref(device), 0)
    
    context = c_void_p()
    cuCtxCreate(byref(context), 0, device)

    # 3. Загрузка скомпилированного модуля и функции (ядра)
    module = c_void_p()
    cuModuleLoad(byref(module), b"matmul.ptx")
    
    kernel = c_void_p()
    cuModuleGetFunction(byref(kernel), module, b"matmul")

    # 4. Выделение памяти на видеокарте (Device)
    d_A = c_void_p()
    d_B = c_void_p()
    d_C = c_void_p()
    
    cuMemAlloc(byref(d_A), size)
    cuMemAlloc(byref(d_B), size)
    cuMemAlloc(byref(d_C), size)

    # 5. Копирование данных с Host на Device
    cuMemcpyHtoD(d_A, h_A.ctypes.data_as(c_void_p), size)
    cuMemcpyHtoD(d_B, h_B.ctypes.data_as(c_void_p), size)

    # 6. Подготовка аргументов для ядра
    N_c = c_int(N)
    kernel_args = (c_void_p * 4)(
        cast(byref(d_A), c_void_p),
        cast(byref(d_B), c_void_p),
        cast(byref(d_C), c_void_p),
        cast(byref(N_c), c_void_p)
    )

    block_size = 16
    grid_size = (N + block_size - 1) // block_size

    # --- НАЧАЛО ЗАМЕРА ВРЕМЕНИ ---
    start_time = time.time()

    # 7. Запуск ядра
    cuLaunchKernel(
        kernel,
        grid_size, grid_size, 1,    
        block_size, block_size, 1,  
        0,                          
        None,                       
        kernel_args,                
        None                        
    )

    cuCtxSynchronize()
    
    # --- КОНЕЦ ЗАМЕРА ВРЕМЕНИ ---
    end_time = time.time()

    # 8. Копирование результата обратно на Host
    cuMemcpyDtoH(h_C.ctypes.data_as(c_void_p), d_C, size)

    # 9. Очистка памяти и контекста
    cuMemFree(d_A)
    cuMemFree(d_B)
    cuMemFree(d_C)
    cuCtxDestroy(context)

    print(f"Размер матриц: {N} x {N}")
    print(f"Время выполнения (CUDA Driver API Python): {end_time - start_time:.5f} секунд")

if __name__ == '__main__':
    main()
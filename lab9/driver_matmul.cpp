#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cuda.h> // Заголовочный файл для Driver API

int main() {
    int N = 512;
    size_t size = N * N * sizeof(float);

    // 1. Подготовка данных на хосте (CPU)
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N, 0.0f);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 2. Инициализация CUDA Driver API
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    
    // Используем современный подход создания контекста (Primary Context)
    CUcontext context;
    cuDevicePrimaryCtxRetain(&context, device);
    cuCtxSetCurrent(context);

    // 3. Загрузка скомпилированного модуля (PTX) и функции (ядра)
    CUmodule module;
    // ВАЖНО: Файл matmul.ptx должен лежать в этой же папке!
    CUresult res = cuModuleLoad(&module, "matmul.ptx");
    if (res != CUDA_SUCCESS) {
        std::cerr << "Ошибка: не удалось загрузить matmul.ptx!" << std::endl;
        return 1;
    }

    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "matmul");

    // 4. Выделение памяти на видеокарте (Device)
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_C, size);

    // 5. Копирование данных с Host на Device
    cuMemcpyHtoD(d_A, h_A.data(), size);
    cuMemcpyHtoD(d_B, h_B.data(), size);

    // 6. Подготовка аргументов для ядра
    // Driver API требует массив указателей на каждый аргумент
    void* args[] = { &d_A, &d_B, &d_C, &N };

    int blockSize = 16;
    int gridSize = (N + blockSize - 1) / blockSize;

    // --- НАЧАЛО ЗАМЕРА ВРЕМЕНИ ---
    auto start_time = std::chrono::high_resolution_clock::now();

    // 7. Запуск ядра
    cuLaunchKernel(kernel,
                   gridSize, gridSize, 1,    // Размеры сетки
                   blockSize, blockSize, 1,  // Размеры блока
                   0, NULL,                  // Разделяемая память и stream
                   args, 0);                 // Аргументы

    // Ожидание завершения работы GPU
    cuCtxSynchronize();

    // --- КОНЕЦ ЗАМЕРА ВРЕМЕНИ ---
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    // 8. Копирование результата обратно на Host
    cuMemcpyDtoH(h_C.data(), d_C, size);

    std::cout << "Размер матриц: " << N << " x " << N << std::endl;
    std::cout << "Время выполнения (CUDA Driver API C/C++): " << diff.count() << " секунд" << std::endl;

    // 9. Очистка памяти и контекста
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    
    // Освобождаем первичный контекст
    cuDevicePrimaryCtxRelease(device);

    return 0;
}
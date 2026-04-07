jit - just in time
компилирует функции и методы при первом вызове. потом кэширует.

ptx - работает с похожим на ассемблер кодом который компилируется с помощью jit
`nvcc --ptx my_kernels.cu -o my_kernels.ptx`

cubin - бинарник, под конкретную архитектуру.
`nvcc -arch=sm_75 -cubin my_kernels.cu -o my_kernels.cubin`

extern "C" - нужен чтобы сохранить имя функции, c++ по умолчанию меняет имена функции для обеспечения работы перегрузок

context (driver api) - изолирует ресурсы, аналог "процесса"

primary context - дает чуть лучшую производительность, в то время как создание собственного контекста

numba - jit компиляция python кода
pycuda - обертка cuda driver api

byref() - аналог &
b"" - bytes literal (ascii 0-255) (в "" используется unicode)

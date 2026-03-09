CC = nvcc
CFLAGS = -arch=sm_75

SRC = simple.cu
SRC1 = local_mem.cu
SRC2 = show.cu

TAR_SIMPL = program.out
TAR_NOT_ENGH = local_mem.out
TAR_JUST_FOR_SHOW = show.out

.PHONY: all clean

all: $(TAR_SIMPL) $(TAR_NOT_ENGH) $(TAR_JUST_FOR_SHOW)

$(TAR_SIMPL): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@

$(TAR_NOT_ENGH): $(SRC1)
	$(CC) $(CFLAGS) --maxrregcount 16 $(SRC1) -o $@ 
# maxrregcout используется для ограничения 
# количества используемых ядром регистров

$(TAR_JUST_FOR_SHOW): $(SRC2)
	$(CC) $(CFLAGS) $(SRC2) -o $@

run:
	./$(TAR_JUST_FOR_SHOW)

# Определите время выполнения соответствующих ядер на
# GPU.
time:
	ncu --metrics gpu__time_duration.sum ./$(TAR_SIMPL)

# Определите для обоих случаев пропускную способность
# при загрузке из глобальной памяти и при сохранении в
# глобальной память.
throughput:
	ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum\
	,gpu__time_duration.sum ./$(TAR_SIMPL)

# Эмулируйте недостаток регистров и определите
# использование локальной памяти.
local:
	ncu --metrics sm__sass_data_bytes_mem_local_op_ld.sum\
	,sm__sass_data_bytes_mem_local_op_st.sum ./$(TAR_NOT_ENGH)

clean:
	rm -rf $(TAR_SIMPL) $(TAR_NOT_ENGH) $(TAR_JUST_FOR_SHOW)
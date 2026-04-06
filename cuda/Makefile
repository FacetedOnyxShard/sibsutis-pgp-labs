CC = nvcc
CFLAGS = -arch=sm_75

SRCS = main.cu

TARGET = program

.PHONY: all run clean

all: $(TARGET)

run:
	./$(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $@

clean:
	rm -rf $(TARGET)


nvcc -arch=sm_75 
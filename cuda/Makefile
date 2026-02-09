CC = nvcc
CFLAGS = -arch=sm_75

SRCS = main.cu

TARGET = program

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $@

clean:
	rm -rf $(TARGET)
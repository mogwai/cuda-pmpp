# Makefile for CUDA program

CC = nvcc
CFLAGS = -std=c++11

all: vectorAdd.bin

vectorAdd.bin: vectorAdd.cu
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f *.bin


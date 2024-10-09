# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = icpc
OPT =
CFLAGS = -Wall -DGETTIMEOFDAY -std=c99 $(OPT) -diag-disable=10441
LDFLAGS = -Wall
# mkl is needed for blas implementation
LDLIBS = -lpthread -lm -diag-disable=10441 -Ofast -xHost -march=native -qopt-report=5 -static-libstdc++ -fno-alias -ffast-math -mavx512f -mavx512dq

targets = benchmark-final benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark.o sgemm-naive.o sgemm-blocked.o sgemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o sgemm-naive.o
	$(CC) -o $@ $^ $(LDLIBS)
### 自己创建的其他算法实习
benchmark-blocked : benchmark.o sgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o sgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

### 最终提交版本 (sgemm-blocked.o修改成对应的文件)
benchmark-final : benchmark.o sgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.cpp
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)

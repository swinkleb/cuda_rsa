NVCC = nvcc
NFLAGS = -O3 -g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
CC = gcc
CFLAGS = -Wall -pedantic -g -c -std=c99
LD = nvcc 
#LDFLAGS = -L/home/clupo/gmp/lib/ -lgmp -lm
LDFLAGS = -lgmp -lm

all: main

gcd.o:
	$(CC) $(CFLAGS) gcd.c

io.o:
	$(CC) $(CFLAGS) io.c

rsa.o:
	$(CC) $(CFLAGS) rsa.c

gcdCuda.o:
	$(NVCC) $(NFLAGS) gcdCuda.cu

main.o:
	$(NVCC) $(NFLAGS) main.c

main: rsa.o io.o gcd.o gcdCuda.o main.o
	$(LD) $(LDFLAGS) main.o gcd.o rsa.o io.o gcdCuda.o -o rsa

clean:
	rm -rf *.o rsa

redo:
	make clean;
	clear;
	make;

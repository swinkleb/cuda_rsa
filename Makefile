NVCC = nvcc
NFLAGS = 
CC = gcc
CFLAGS = -Wall -pedantic -g -c -std=c99
LD = nvcc
#LDFLAGS = -L/home/clupo/gmp/lib/ -lgmp -lm
LDFLAGS = -lgmp

all: main

gcd.o:
	$(CC) $(CFLAGS) gcd.c

io.o:
	$(CC) $(CFLAGS) io.c

rsa.o:
	$(CC) $(CFLAGS) rsa.c

gcdCuda.o:
	$(NVCC) $(NFLAGS) rsaCuda.cu

main.o:
	$(CC) $(CFLAGS) main.c

main: main.o rsa.o io.o gcd.o
	$(LD) $(LDFLAGS) main.o gcd.o rsa.o io.o -o rsa

clean:
	rm -rf *.o rsa

CC = gcc
CFLAGS = -Wall -pedantic -g -c -std=c99
LD = gcc
LDFLAGS = -L/home/clupo/gmp/lib/ -lgmp
#LDFLAGS = -lgmp

all: main

gcd.o:
	$(CC) $(CFLAGS) gcd.c

io.o:
	$(CC) $(CFLAGS) io.c

rsa.o:
	$(CC) $(CFLAGS) rsa.c

main.o:
	$(CC) $(CFLAGS) main.c

main: main.o rsa.o io.o gcd.o
	$(LD) $(LDFLAGS) main.o gcd.o rsa.o io.o -o rsa

clean:
	rm -rf *.o rsa

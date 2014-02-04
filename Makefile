# Scott Winkleblack
# Assignment 1
# csc 556

CC = gcc
CFLAGS = -Wall -pedantic -g -c
LD = gcc
LDFLAGS = -lgmp

all: rsa gcd

gcd:

io: io.o
	$(CC) $(CFLAGS) io.c

rsa.o:
	$(CC) $(CFLAGS) rsa.c

rsa: rsa.o io.o
	$(LD) $(LDFLAGS) rsa.o io.o -o rsa

clean:
	rm -rf *.o gcd rsa

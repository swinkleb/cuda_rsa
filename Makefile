# Scott Winkleblack
# Assignment 1
# csc 556

CC = gcc
CFLAGS = -Wall -pedantic -g -c
LD = gcc
LDFLAGS = -lgmp

all: rsa

gcd: gcd.o
	$(CC) $(CFLAGS) gcd.c

io: io.o
	$(CC) $(CFLAGS) io.c

rsa.o:
	$(CC) $(CFLAGS) rsa.c

rsa: rsa.o io.o gcd.o
	$(LD) $(LDFLAGS) gcd.o rsa.o io.o -o rsa

clean:
	rm -rf *.o rsa

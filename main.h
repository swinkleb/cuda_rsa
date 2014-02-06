#include "stdint.h"

#define MIN_ARG_COUNT 3
#define PROG_ARG 0
#define FLAG_ARG 1
#define IN_FILE_ARG 2
#define OUT_FILE_ARG 3
#define DEFAULT_OUT_FILE "output.txt"

#ifndef NUM_INTS
#define NUM_INTS 32
#endif

typedef struct u1024bit_t{
       uint32_t number[NUM_INTS];
} u1024bit_t;

void usage(char *this);

void cpuImpl(char *inFile, char *outFile);

void gpuImpl(char *inFile, char *outFIle);

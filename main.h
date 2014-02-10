#include "stdint.h"
#include "math.h"
#include "stdlib.h"

#define MIN_ARG_COUNT 3
#define PROG_ARG 0
#define FLAG_ARG 1
#define IN_FILE_ARG 2
#define D_OUT_FILE_ARG 3 
#define N_OUT_FILE_ARG 4
#define DEFAULT_D_OUT_FILE "outputD.txt"
#define DEFAULT_N_OUT_FILE "outputN.txt"
#define DEFAULT_OUT_FILE "output.txt"
#define NUM_INTS 32

// block and grid dimensions
#define BLOCK_DIM_Y 8 // be careful changing this; bit vector size depends on it
#define BLOCK_DIM_X NUM_INTS
#define GRID_DIM_X 23
#define GRID_DIM_Y 1
#define NUM_BLOCKS (GRID_DIM_X * GRID_DIM_Y)

typedef struct u1024bit_t {
       uint32_t number[NUM_INTS];
} u1024bit_t;

void usage(char *myName);

void cpuImpl(char *inFile, char *outFile);

void gpuImpl(char *inFile, char *outFile);

void testImpl(char *inFile, char *outFile);

void print1024Int(uint32_t *number);

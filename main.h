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

// block and grid dimensions
#define BLOCK_DIM_Y 8 // be careful changing this; bit vector size depends on it
#define BLOCK_DIM_X NUM_INTS
#define GRID_DIM_X 2
#define GRID_DIM_Y 1
#define NUM_BLOCKS GRID_DIM_X * GRID_DIM_Y

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef struct u1024bit_t{
       uint32_t number[NUM_INTS];
} u1024bit_t;

void usage(char *this);

void cpuImpl(char *inFile, char *outFile);

void gpuImpl(char *inFile, char *outFIle);

static void HandleError( cudaError_t err, const char *file, int line);
#include "gcd.h"
#include "main.h"
#include "io.h"

int main (int argc, char **argv)
{
   if (argc < MIN_ARG_COUNT)
   {   
      usage(argv[PROG_ARG]);
   }   

   switch (argv[FLAG_ARG][0])
   {   
      case 'c':
         cpuImpl(argv[IN_FILE_ARG], argc > MIN_ARG_COUNT ? argv[OUT_FILE_ARG] : NULL);
         break;

      case 'g':
         gpuImpl(argv[IN_FILE_ARG], argc > MIN_ARG_COUNT ? argv[OUT_FILE_ARG] : NULL);
         break;

      default:
         usage(argv[PROG_ARG]);
   }   
}

void usage(char *this)
{
   printf("Usage: %s <flag> <input_file> [output_file]\n", this);
   printf("Flags:\n");
   printf("\tc - cpu implementation\n");
   printf("\tg - gpu implementation\n");

   exit(1);
}

void cpuImpl(char *inFile, char *outFile)
{
   mpz_t *array;
   unsigned int count;

   count = readKeysFromFileMPZ(&array, inFile);
   count = findGCDs(array, count, outFile == NULL ? DEFAULT_OUT_FILE : outFile);
   
   printf("Total number of bad keys found: %d\n", count);
}

void gpuImpl(char *inFile, char *outFile)
{
   u1024bit_t *array;
//   u1024bit_t *privateKeys;
   unsigned int count;

   count = readKeysFromFile(&array, inFile);

   // do GCDs
   // resultant bit vector on host
   uint8_t bitVector[NUM_BLOCKS];

   // pointers on device
   u1024bit_t *d_keys;
   u1024bit_t *d_currentKey;
   uint8_t *d_bitVector;

   dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
   dim3 gridDim(GRID_DIM_X, GRID_DIM_Y);

   // allocate space for current key, keys to compare and bit vector
   HANDLE_ERROR(cudaMalloc((void **) &d_currentKey,
      sizeof(u1024bit_t)));
   HANDLE_ERROR(cudaMalloc((void **) &d_keys,
      sizeof(u1024bit_t) * BLOCK_DIM_Y * NUM_BLOCKS));
   HANDLE_ERROR(cudaMalloc((void **) &d_bitVector,
      sizeof(uint8_t) * NUM_BLOCKS);

   int i, j;
   int toCopy;

   for (i = 0; i < count; i++) {
      for (j = i + 1; j < count; j += NUM_BLOCKS) {
         // copy current key
         HANDLE_ERROR(cudaMemcpy(d_currentKey, array + i,
            sizeof(u1024bit_t),
            cudaMemcpyHostToDevice));

         // copy list of keys
         toCopy = j + NUM_BLOCKS >= count ?
             (count - j) * BLOCK_DIM_Y : BLOCK_DIM_Y * NUM_BLOCKS;

         // add a comment here explaining this
         HANDLE_ERROR(cudaMemset(d_keys, 0,
            sizeof(u1024bit_t) * BLOCK_DIM_Y * NUM_BLOCKS));

         HANDLE_ERROR(cudaMemcpy(d_keys, array + j,
            sizeof(u1024bit_t) * toCopy,
            cudaMemcpyHostToDevice));

         // initialize bit vector to 0
         HANDLE_ERROR(cudaMemset(d_bitVector, 0,
            sizeof(uint8_t) * NUM_BLOCKS));

         // kernel call
         cuGCD<<<gridDim, blockDim>>>(d_currentKey, d_keys, d_bitVector);
         HANDLE_ERROR(cudaPeekAtLastError());

         // copy bit vector back
         HANDLE_ERROR(cudaMemcpy(bitVector, d_bitVector,
            sizeof(uint8_t) * NUM_BLOCKS,
            cudaMemcpyDeviceToHost));

         outputFunc(i, j, bitVector);
      }
   }

   // do freeing
   cudaFree(d_keys);
   cudaFree(d_currentKey);
   cudaFree(d_bitVector);

   // look through bit vector

   outputKeysToFile(array, count, outFile == NULL ? DEFAULT_OUT_FILE : outFile);
}

static void HandleError( cudaError_t err,
    const char *file,
    int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
    exit( EXIT_FAILURE );
  }
}






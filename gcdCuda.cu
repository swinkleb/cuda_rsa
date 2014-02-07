#include "gmp_location.h"
#include "main.h"
#include "io.h"
#include "gcd.h"
#include "gcdCuda.h"

void dispatchGcdCalls(u1024bit_t *array, uint32_t *found, int count, FILE *dfp, FILE *nfp) {

   // prints out inputted list of numbers
   /*
   printf("In dispatchGcdCalls\n");
   printf("Count: %i\n", count);

   int ugh;
   for (ugh = 0; ugh < count; ugh++) {
      print1024Int(array[ugh].number);
   }
   */

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
      sizeof(uint8_t) * NUM_BLOCKS));

   int i;
   int j;
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

         computeAndOutputGCDs(array, found, bitVector, i, j, dfp, nfp);
      }
   }

   // do freeing
   cudaFree(d_keys);
   cudaFree(d_currentKey);
   cudaFree(d_bitVector);
}

__global__ void cuGCD(u1024bit_t *key, u1024bit_t *key_comparison_list, 
   uint8_t *bitvector) {

    /*We are using blocks of size (x, y) (32, 6),
    so each row in a block will be responsible for computing one set of
    key comparisons*/

    /*OLD*/
   /*int keyNum = blockIdx.y * gridDim.x + blockIdx.x;*/

   /*New*/
   int keyNum = (blockIdx.y * gridDim.x + blockIdx.x) * 
        (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x);
 
   // make this prettier
   int i = 0;
   __shared__ u1024bit_t shkey[BLOCK_DIM_Y];

   int j;
   for (j = 0; j < BLOCK_DIM_Y; j++) {

      for (i = 0; i < NUM_INTS; i++){

         shkey[j].number[i] = key->number[i];
      }
   }

   gcd(shkey[blockIdx.y].number, key_comparison_list[keyNum].number);

   if (isGreaterThanOne(key_comparison_list[keyNum].number)) {
      (bitvector[blockIdx.y * gridDim.x + blockIdx.x]) |=
         (LOW_ONE_MASK << threadIdx.y);
   }
}

// result ends up in y; x is also overwritten
__device__ void gcd(unsigned int *x, unsigned int *y) {
   int c = 0;

   // __syncthreads(); // definitely needed here

   // we think this loop is okay
   while (((x[WORDS_PER_KEY - 1] | y[WORDS_PER_KEY - 1]) & 1) == 0) {
      shiftR1(x);
      shiftR1(y);
      c++;
   }

   while (isNonZero(x)) {

      while ((x[WORDS_PER_KEY - 1] & 1) == 0) {
         shiftR1(x);
      }

      // SOMETHING BAD HAPPENS AROUND HERE

      while ((y[WORDS_PER_KEY - 1] & 1) == 0) {
         return;
         shiftR1(y);
      }

      if (geq(x, y)) {
         subtract(x, y);
         shiftR1(x);
      }
      else {
         subtract(y, x);
         shiftR1(y);
      }
   }

   // __syncthreads(); // definitely needed here

   shiftL(y, c);
}

__device__ void shiftR1(unsigned int *arr)
{
   unsigned int index = threadIdx.x;
   uint32_t temp;

   if (index != 0)
   {
      temp = arr[index - 1];
   }
   else
   {
      temp = 0;
   }

   arr[index] >>= 1;
   arr[index] |= (temp << WORDS_PER_KEY - 1);
}

__device__ void shiftL1(unsigned int *arr)
{
   unsigned int index = threadIdx.x;
   uint32_t temp;

   if (index != WORDS_PER_KEY - 1)
   {
      temp = arr[index + 1];
   }
   else
   {
      temp = 0;
   }

   arr[index] <<= 1;
   arr[index] |= (temp >> WORDS_PER_KEY - 1);
}

__device__ void shiftL(unsigned int *arr, unsigned int x) {
   int i;
   for (i = 0; i < x; i++) {
      shiftL1(arr);
   }
}

__device__ void subtract(uint32_t *x, uint32_t *y) {
   __shared__ uint8_t borrow[BLOCK_DIM_Y][WORDS_PER_KEY];

   uint8_t index = threadIdx.x;

   // initialize borrow array to 0
   borrow[threadIdx.y][index] = 0;

   if (x[index] < y[index] && index > 0) {
      borrow[threadIdx.y][index - 1] = 1;
   }

   x[index] = x[index] - y[index];

   int underflow = 0;

   while (__any(borrow[threadIdx.y][index])) {
      if (borrow[threadIdx.y][index]) {
         underflow = x[index] < 1;
         x[index] = x[index] - 1;

         if (underflow && index > 0) {
            borrow[threadIdx.y][index - 1] = 1;
         }

         borrow[threadIdx.y][index] = 0;
      }
   }
}

__device__ int geq(uint32_t *x, uint32_t *y) {
   __shared__ unsigned int pos[BLOCK_DIM_Y];

   int index = threadIdx.x;

   if (index == 0) {
      pos[threadIdx.y] = WORDS_PER_KEY - 1;
   }

   if (x[index] != y[index]) {
      atomicMin(&pos[threadIdx.y], index);
   }

   return x[pos[threadIdx.y]] >= y[pos[threadIdx.y]];
}

__device__ int isNonZero(uint32_t *x) {
   __shared__ uint8_t nonZeroFound[BLOCK_DIM_Y];

   uint8_t index = threadIdx.x;

   if (index == 0) {
      nonZeroFound[threadIdx.y] = 0;
   }

   if (x[index] != 0) {
      nonZeroFound[threadIdx.y] = 1;
   }

   return nonZeroFound[threadIdx.y];
}

__device__ int isGreaterThanOne(uint32_t *number) {
   __shared__ uint8_t greaterThanOne[BLOCK_DIM_Y];

   uint8_t index = threadIdx.x;

   if (index == 0) {
      greaterThanOne[threadIdx.y] = 0;
   }

   if (index < WORDS_PER_KEY - 1 && number[index] > 0) {
      greaterThanOne[threadIdx.y] = 1;
   }
   else if (index == WORDS_PER_KEY - 1 && number[index] > 1) {
      greaterThanOne[threadIdx.y] = 1;
   }

   return greaterThanOne[threadIdx.y];
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

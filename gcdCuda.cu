#include "gcdCuda.h"
#include "io.h"
#include "main.h"

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
 
   int i = 0;
   __shared__ u1024bit_t shkey;

   for (i = 0; i < NUM_INTS; i++){

      shkey.number[i] = key.number[i];
   }

   __syncthreads();

   gcd(shkey->number, key_comparison_list[keyNum]->number);

   if (isGreaterThanOne(key_comparison_list[keyNum]->number)) {
      (bitvector[blockIdx.y * gridDim.x + blockIdx.x]) |=
         (LOW_ONE_MASK << threadIdx.y);
   }
}

// result ends up in y; x is also overwritten
// had to mess around with __syncthreads() in different places;
// notes from testing are still present
__device__ void gcd(unsigned int *x, unsigned int *y) {
   int c = 0;

   __syncthreads(); // definitely needed here

   while (((x[WORDS_PER_KEY - 1] | y[WORDS_PER_KEY - 1]) & 1) == 0) {
      shiftR1(x);
      shiftR1(y);
      c++;
   }

   while (isNonZero(x)) {

      while ((x[WORDS_PER_KEY - 1] & 1) == 0) {
         shiftR1(x);
      }

      while ((y[WORDS_PER_KEY - 1] & 1) == 0) {
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

   __syncthreads(); // definitely needed here

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
      __syncthreads(); // OLD
   }
}

__device__ void subtract(uint32_t *x, uint32_t *y) {
   __shared__ uint8_t borrow[BLOCK_DIM_Y][WORDS_PER_KEY];

   uint8_t index = threadIdx.x;

   // initialize borrow array to 0
   borrow[threadIdx.y][index] = 0;
   __syncthreads(); // OLD

   if (x[index] < y[index] && index > 0) {
      borrow[threadIdx.y][index - 1] = 1;
   }

   x[index] = x[index] - y[index];

   int underflow = 0;

   __syncthreads(); // OLD

   while (__any(borrow[threadIdx.y][index])) {
      if (borrow[threadIdx.y][index]) {
         underflow = x[index] < 1;
         x[index] = x[index] - 1;

         if (underflow && index > 0) {
            borrow[threadIdx.y][index - 1] = 1;
         }

         borrow[threadIdx.y][index] = 0;
      }
      __syncthreads(); // OLD
   }
}

__device__ int geq(uint32_t *x, uint32_t *y) {
   __shared__ unsigned int pos[BLOCK_DIM_Y];

   int index = threadIdx.x;

   if (index == 0) {
      pos[threadIdx.y] = WORDS_PER_KEY - 1;
   }
   __syncthreads(); // OLD

   if (x[index] != y[index]) {
      atomicMin(&pos[threadIdx.y], index);
   }

   __syncthreads(); // OLD
   return x[pos[threadIdx.y]] >= y[pos[threadIdx.y]];
}

__device__ int isNonZero(uint32_t *x) {
   __shared__ uint8_t nonZeroFound[BLOCK_DIM_Y];

   uint8_t index = threadIdx.x;

   if (index == 0) {
      nonZeroFound[threadIdx.y] = 0;
   }
   __syncthreads(); // OLD

   if (x[index] != 0) {
      nonZeroFound[threadIdx.y] = 1;
   }

   __syncthreads(); // OLD
   return nonZeroFound[threadIdx.y];
}

__device__ int isGreaterThanOne(uint32_t *number) {
   __shared__ uint8_t greaterThanOne[BLOCK_DIM_Y];

   uint8_t index = threadIdx.x;

   if (index == 0) {
      greaterThanOne[threadIdx.y] = 0;
   }
   __syncthreads(); // OLD

   if (index < WORDS_PER_KEY - 1 && number[index] > 0) {
      greaterThanOne[threadIdx.y] = 1;
   }
   else if (index == WORDS_PER_KEY - 1 && number[index] > 1) {
      greaterThanOne[threadIdx.y] = 1;
   }

   __syncthreads(); // OLD
   return greaterThanOne[threadIdx.y];
}

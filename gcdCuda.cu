#include "gcdCuda.h"
#include "io.h"

// result ends up in y; x is also overwritten
__device__ void gcd(unsigned int *x, unsigned int *y) {
   int c = 0;

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
      __syncthreads();
   }
}

__device__ void subtract(uint32_t *x, uint32_t *y) {
   __shared__ uint8_t borrow[WORDS_PER_KEY];

   uint8_t index = threadIdx.x;

   // initialize borrow array to 0
   borrow[index] = 0;
   __syncthreads();

   if (x[index] < y[index] && index > 0) {
      borrow[index - 1] = 1;
   }

   x[index] = x[index] - y[index];

   int underflow = 0;

   __syncthreads();
   while (__any(borrow[index])) {
      if (borrow[index]) {
         underflow = x[index] < 1;
         x[index] = x[index] - 1;

         if (underflow && index > 0) {
            borrow[index - 1] = 1;
         }

         borrow[index] = 0;
      }
      __syncthreads();
   }
}

__device__ int geq(uint32_t *x, uint32_t *y) {
   __shared__ uint8_t pos;

   uint8_t index = threadIdx.x;

   if (index == 0) {
      pos = WORDS_PER_KEY - 1;
   }
   __syncthreads();

   if (x[index] != y[index]) {
      atomicMin((int *) &pos, (int) index);
   }

   __syncthreads();
   return x[pos] >= y[pos];
}

__device__ int isNonZero(uint32_t *x) {
   __shared__ uint8_t nonZeroFound;

   uint8_t index = threadIdx.x;

   if (index == 0) {
      nonZeroFound = 0;
   }
   __syncthreads();

   if (x[index] != 0) {
      nonZeroFound = 1;
   }

   __syncthreads();
   return nonZeroFound;
}


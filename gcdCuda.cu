#include "gmp_location.h"
#include "main.h"
#include "io.h"
#include "gcd.h"
#include "gcdCuda.h"

void dispatchGcdCalls(u1024bit_t *array, uint32_t *found, int count, FILE *dfp, FILE *nfp) {

   // resultant bit vector on host
   u1024bit_t *pinnedArray;
   uint8_t *bitVector_s1;
   uint8_t *bitVector_s2;
   
   HANDLE_ERROR(cudaMallocHost(&pinnedArray, sizeof(u1024bit_t) * count));
   memcpy(pinnedArray, array, sizeof(u1024bit_t) * count);

   HANDLE_ERROR(cudaMallocHost(&bitVector_s1, sizeof(uint8_t) * NUM_BLOCKS));
   HANDLE_ERROR(cudaMallocHost(&bitVector_s2, sizeof(uint8_t) * NUM_BLOCKS));

   // pointers on device
   u1024bit_t *d_currentKey;
   u1024bit_t *d_keys_s1;
   uint8_t *d_bitVector_s1;
   u1024bit_t *d_keys_s2;
   uint8_t *d_bitVector_s2;

   // stream stuff
   cudaStream_t s1;
   HANDLE_ERROR(cudaStreamCreate(&s1));
   cudaStream_t s2;
   HANDLE_ERROR(cudaStreamCreate(&s2));

   cudaEvent_t temp_event;
   cudaEvent_t active_s1;
   HANDLE_ERROR(cudaEventCreate(&active_s1));
   cudaEvent_t active_s2;
   HANDLE_ERROR(cudaEventCreate(&active_s2));
   cudaEvent_t passive_s1;
   HANDLE_ERROR(cudaEventCreate(&passive_s1));
   cudaEvent_t passive_s2;
   HANDLE_ERROR(cudaEventCreate(&passive_s2));

   // allocate space for current key, keys to compare and bit vector
   HANDLE_ERROR(cudaMalloc((void **) &d_currentKey,
      sizeof(u1024bit_t)));

   HANDLE_ERROR(cudaMalloc((void **) &d_keys_s1,
      sizeof(u1024bit_t) * BLOCK_DIM_Y * NUM_BLOCKS));
   HANDLE_ERROR(cudaMalloc((void **) &d_bitVector_s1,
      sizeof(uint8_t) * NUM_BLOCKS));
   HANDLE_ERROR(cudaMalloc((void **) &d_keys_s2,
      sizeof(u1024bit_t) * BLOCK_DIM_Y * NUM_BLOCKS));
   HANDLE_ERROR(cudaMalloc((void **) &d_bitVector_s2,
      sizeof(uint8_t) * NUM_BLOCKS));

   int i;
   int j;
   int j1;
   int j2;
   int stride = NUM_BLOCKS * BLOCK_DIM_Y;

   for (i = 0; i < count; i++) {
      for (j = i + 1; j < count; j += stride) {
         // copy current key
         HANDLE_ERROR(cudaMemcpy(d_currentKey, array + i,
            sizeof(u1024bit_t),
            cudaMemcpyHostToDevice));

         j1 = j;
         callCudaStreams(pinnedArray, bitVector_s1, d_keys_s1, d_currentKey, d_bitVector_s1,
               &j, count, stride, s1, active_s1);
         j2 = j;
         callCudaStreams(pinnedArray, bitVector_s2, d_keys_s2, d_currentKey, d_bitVector_s2,
               &j, count, stride, s2, active_s2);
         
         cudaEventSynchronize(active_s1);
         computeAndOutputGCDs(array, found, bitVector_s1, i, j1, dfp, nfp);
         cudaEventSynchronize(active_s2);
         computeAndOutputGCDs(array, found, bitVector_s2, i, j2, dfp, nfp);
      }
   }

   // do freeing
   cudaFree(d_currentKey);

   cudaFree(d_keys_s1);
   cudaFree(d_bitVector_s1);
   cudaFree(d_keys_s2);
   cudaFree(d_bitVector_s2);
}

void callCudaStreams(u1024bit_t *array, uint8_t *bitVector,
      u1024bit_t *d_keys, u1024bit_t *d_currentKey, uint8_t *d_bitVector,
      int *j, int count, int stride, cudaStream_t stream, cudaEvent_t event)
{
   static dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
   static dim3 gridDim(GRID_DIM_X, GRID_DIM_Y);

   // copy list of keys
   int toCopy = *j + stride >= count ? count - *j : stride;
   if (toCopy < 0)
   {
      return;
   }

   HANDLE_ERROR(cudaMemcpyAsync(d_keys, array + *j,
            sizeof(u1024bit_t) * toCopy,
            cudaMemcpyHostToDevice, stream));

   // initialize bit vector to 0
   HANDLE_ERROR(cudaMemset(d_bitVector, 0,
            sizeof(uint8_t) * NUM_BLOCKS));

   // kernel call
   cuGCD<<<gridDim, blockDim, 0, stream>>>(d_currentKey, d_keys, d_bitVector);

   HANDLE_ERROR(cudaPeekAtLastError());

   // copy bit vector back
   HANDLE_ERROR(cudaMemcpyAsync(bitVector, d_bitVector,
            sizeof(uint8_t) * NUM_BLOCKS,
            cudaMemcpyDeviceToHost, stream));

   cudaEventRecord(event, stream);
   
   *j = *j + stride;
}

__global__ void cuGCD(u1024bit_t *key, u1024bit_t *key_comparison_list, 
   uint8_t *bitvector) {

   __shared__ u1024bit_t shkey[BLOCK_DIM_Y * GRID_DIM_X];

    /* We are using blocks of size (x, y) (32, 8),
    so each row in a block will be responsible for computing one set of
    key comparisons */

   int keyNum = (BLOCK_DIM_Y * blockIdx.x) + threadIdx.y;
   int index = threadIdx.x;

   int i;
   for (i = 0; i < BLOCK_DIM_Y * GRID_DIM_X; i++) {
      shkey[i].number[index] = key->number[index];
   }

   __syncthreads();

   gcd(shkey[keyNum].number, key_comparison_list[keyNum].number);

   if (isGreaterThanOne(key_comparison_list[keyNum].number)) {
      bitvector[keyNum / 8] |= LOW_ONE_MASK << (keyNum % 8);
   }
}

// result ends up in y; x is also overwritten
__device__ void gcd(unsigned int *x, unsigned int *y) {
   int c = 0;

   if (isNonZero(x) && isNonZero(y)) {

      while (((x[WORDS_PER_KEY - 1] | y[WORDS_PER_KEY - 1]) & 1) == 0) {
         shiftR1(x);
         shiftR1(y);
         c++;
      }

      while (__any(x[threadIdx.x])) {

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
   else if (isNonZero(y)) {
      y[threadIdx.x] = x[threadIdx.x];
   }
}

__device__ void shiftR1(unsigned int *arr)
{
   unsigned int index = threadIdx.x;
   uint32_t temp = 0;

   if (index != 0)
   {
      temp = arr[index - 1];
   }

   arr[index] >>= 1;
   arr[index] |= (temp << WORDS_PER_KEY - 1);
}

__device__ void shiftL1(unsigned int *arr)
{
   unsigned int index = threadIdx.x;
   uint32_t temp = 0;

   if (index != WORDS_PER_KEY - 1)
   {
      temp = arr[index + 1];
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
   uint8_t *borrowPtr = borrow[threadIdx.y];

   uint8_t index = threadIdx.x;

   if (index == 0) {
      borrowPtr[WORDS_PER_KEY - 1] = 0;
   }

   unsigned int temp;
   temp = x[index] - y[index];

   if (index > 0) {
      borrowPtr[index - 1] = (temp > x[index]);
   }

   while (__any(borrowPtr[index])) {
      if (borrowPtr[index]) {
         temp--;
      }

      if (index > 0) {
         borrowPtr[index - 1] = (temp == 0xffffffffU && borrow[index]);
      }
   }

   x[index] = temp;
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

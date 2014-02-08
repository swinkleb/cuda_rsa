#include "gmp_location.h"
#include "main.h"
#include "io.h"
#include "gcd.h"
#include "gcdCuda.h"
#define FATAL(msg, ...) \
  do {\
    fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
    exit(-1);\
  } while(0)


void dispatchGcdCalls(u1024bit_t *array, uint32_t *found, int count, FILE *dfp, FILE *nfp) {

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
   int stride = NUM_BLOCKS * BLOCK_DIM_Y;

    cudaError_t cuda_ret;
    cudaStream_t cuda_stream1;
    cudaStream_t cuda_stream2;
    size_t pending, transfer_size;
    cudaEvent_t memcpy_event, kernel_event;

    cuda_ret = cudaStreamCreate(&cuda_stream1);
    if(cuda_ret != cudaSuccess) FATAL("Unable to create CUDA stream");
    cuda_ret = cudaStreamCreate(&cuda_stream2);
    if(cuda_ret != cudaSuccess) FATAL("Unable to create CUDA stream");

/* Create CUDA events */
        cuda_ret = cudaEventCreate(&memcpy_event);
        if(cuda_ret != cudaSuccess) FATAL("Unable to create CUDA event");
        cuda_ret = cudaEventCreate(&kernel_event);
        if(cuda_ret != cudaSuccess) FATAL("Unable to create CUDA event");

   for (i = 0; i < count; i++) {
      for (j = i + 1; j < count; j += stride) {
         
         // copy current key
         HANDLE_ERROR(cudaMemcpyAsync(d_currentKey, array + i,
            sizeof(u1024bit_t),
            cudaMemcpyHostToDevice, cuda_stream1));

         // copy list of keys
         toCopy = j + stride >= count ? count - j : stride;

         HANDLE_ERROR(cudaMemcpyAsync(d_keys, array + j,
            sizeof(u1024bit_t) * toCopy,
            cudaMemcpyHostToDevice, cuda_stream1));
        
                 // initialize bit vector to 0
         HANDLE_ERROR(cudaMemsetAsync(d_bitVector, 0,
            sizeof(uint8_t) * NUM_BLOCKS, cuda_stream1));

        cuda_ret = cudaEventRecord(memcpy_event, cuda_stream1);
        if(cuda_ret != cudaSuccess) FATAL("UNable to queue CUDA event");
         // kernel call
         cuGCD<<<gridDim, blockDim, 0, cuda_stream1>>>(d_currentKey, d_keys, d_bitVector);

        cuda_ret = cudaEventRecord(kernel_event, cuda_stream1);

        if(cuda_ret != cudaSuccess) FATAL("UNable to queue CUDA event");

         HANDLE_ERROR(cudaPeekAtLastError());

         // copy bit vector back
         HANDLE_ERROR(cudaMemcpyAsync(bitVector, d_bitVector,
            sizeof(uint8_t) * NUM_BLOCKS,
            cudaMemcpyDeviceToHost, cuda_stream1));

        cuda_ret = cudaEventRecord(memcpy_event, cuda_stream1);
        if(cuda_ret != cudaSuccess) FATAL("UNable to queue CUDA event");

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

#include "gmp_location.h"
#include "main.h"
#include "io.h"
#include "gcd.h"
#include "gcdCuda.h"

void testGcdCalls(u1024bit_t *array, uint32_t *found, int count, FILE *dfp, FILE *nfp) {

   uint32_t *n1;
   uint32_t *n2;
   uint32_t factor[32];

   dim3 blockDim(32, 1);
   dim3 gridDim(1, 1);

   mpz_t a;
   mpz_t b;
   mpz_t temp;
   mpz_inits(a, b, temp, NULL);

   // allocate space for current keys
   HANDLE_ERROR(cudaMalloc((void **) &n1,
      sizeof(uint32_t) * 32));
   HANDLE_ERROR(cudaMalloc((void **) &n2,
      sizeof(uint32_t) * 32));

   // copy current keys
   HANDLE_ERROR(cudaMemcpy(n1, array,
      sizeof(uint32_t) * 32,
      cudaMemcpyHostToDevice));

   HANDLE_ERROR(cudaMemcpy(n2, array + 1,
      sizeof(uint32_t) * 32,
      cudaMemcpyHostToDevice));
/*
   mpz_import(temp, 32, 1, 4, 0, 0, 
      array);
   fprintf(stdout, "xxxxx n1:\n");
   mpz_out_str(stdout, BASE_10, temp);
   fprintf(stdout, "\nxxxxx\n");

   mpz_import(temp, 32, 1, 4, 0, 0, 
      array + 1);
   fprintf(stdout, "xxxxx n2:\n");
   mpz_out_str(stdout, BASE_10, temp);
   fprintf(stdout, "\nxxxxx\n");
*/
   // kernel call
   mygcd<<<gridDim, blockDim>>>(n1, n2);
   HANDLE_ERROR(cudaPeekAtLastError());

   HANDLE_ERROR(cudaMemcpy(factor, n2,
      sizeof(uint32_t) * 32,
      cudaMemcpyDeviceToHost));
   printf("%u\n%u\n%u\n", factor[29], factor[30], factor[31]);

   
   fprintf(stdout, "xxxxx p:\n");
   mpz_import(a, 32, 1, 4, 0, 0, 
      array);
   mpz_import(b, 32, 1, 4, 0, 0, 
      array + 1);
   mpz_gcd(temp, a, b);
   mpz_out_str(stdout, BASE_10, temp);
    
   mpz_import(temp, 32, 1, 4, 0, 0, 
      factor);
   fprintf(stdout, "\nxxxxx p:\n");
   mpz_out_str(stdout, BASE_10, temp);
   fprintf(stdout, "\nxxxxx\n");

   // do freeing
   cudaFree(n1);
   cudaFree(n2);
}

__global__ void mygcd(unsigned *x, unsigned *y) {
   int c = 0;
   int tid = threadIdx.x;

   while(((x[32 - 1] | y[32 - 1]) & 1) == 0) {
      shiftR1(x);
      shiftR1(y);
      c++;
   }

   while(__any(x[tid])) {
      while((x[32 - 1] & 1) == 0)
         shiftR1(x);

      while((y[32 - 1] & 1) == 0)
         shiftR1(y);

      if(geq(x, y)) {
         subtract(x, y, x);
         shiftR1(x);
      } else {
         subtract(y, x, y);
         shiftR1(y);
      }
   }

   for(int i = 0; i < c; i++)
      shiftL1(y);
}

__device__ void gcd(unsigned *x, unsigned *y) {
   int c = 0;
   int tid = threadIdx.x;

   while(((x[32 - 1] | y[32 - 1]) & 1) == 0) {
      shiftR1(x);
      shiftR1(y);
      c++;
   }

   while(__any(x[tid])) {
      while((x[32 - 1] & 1) == 0)
         shiftR1(x);

      while((y[32 - 1] & 1) == 0)
         shiftR1(y);

      if(geq(x, y)) {
         subtract(x, y, x);
         shiftR1(x);
      } else {
         subtract(y, x, y);
         shiftR1(y);
      }
   }

   for(int i = 0; i < c; i++)
      shiftL1(y);
}

__device__ void shiftR1(unsigned *x) {
   unsigned x1 = 0;
   int tid = threadIdx.x;

   if(tid)
      x1 = x[tid - 1];

   x[tid] = (x[tid] >> 1) | (x1 << 31);
}

__device__ void shiftL1(unsigned *x) {
   unsigned x1 = 0;
   int tid = threadIdx.x;

   if(tid != 32 - 1)
      x1 = x[tid + 1];

   x[tid] = (x[tid] << 1) | (x1 >> 31);
}

__device__ int geq(unsigned *x, unsigned *y) {
   // pos is the maximum index (which int in the key) where the values are not the same
   __shared__ unsigned int pos[BLOCK_DIM_Y];
   int tid = threadIdx.x;

   if(tid== 0)
      pos[threadIdx.y] = 32 - 1;

   if(x[tid] != y[tid])
      atomicMin(&pos[threadIdx.y], tid);

   return x[pos[threadIdx.y]] >= y[pos[threadIdx.y]];
}

__device__ void subtract(unsigned *x, unsigned *y, unsigned *z) {
   __shared__ unsigned char s_borrow[BLOCK_DIM_Y][32];
   // borrow points to j
   unsigned char *borrow = s_borrow[threadIdx.y];
   int tid = threadIdx.x;

   if(tid == 0)
      borrow[32 - 1] = 0;

   unsigned int t;
   t = x[tid] - y[tid];

   if(tid)
      borrow[tid - 1] = (t > x[tid]);

   while(__any(borrow[tid]))
   {
      if(borrow[tid]) {
         t--;
      }

      if(tid)
         borrow[tid - 1] = (t == 0xffffffffU);
      //borrow[tid - 1] = (t == 0xffffffffU && borrow[tid]);
   }

   z[tid] = t;
}

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

   for (i = 0; i < count; i++) {
      for (j = i + 1; j < count; j += stride) {
         // copy current key
         HANDLE_ERROR(cudaMemcpy(d_currentKey, &array[i],
            sizeof(u1024bit_t),
            cudaMemcpyHostToDevice));

         // copy list of keys
         toCopy = j + stride >= count ? 
            (count - j) * BLOCK_DIM_Y : stride;

         // add a comment here explaining this-- necessary?
         HANDLE_ERROR(cudaMemset(d_keys, 0,
            sizeof(u1024bit_t) * stride));

         HANDLE_ERROR(cudaMemcpy(d_keys, &array[j],
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

   printf("Keynum %i:\n", keyNum);

   gcd(&(shkey[blockIdx.y].number[0]), &(key_comparison_list[keyNum].number[0]));

   if (isGreaterThanOne(key_comparison_list[keyNum].number)) {
      bitvector[keyNum / 8] |= LOW_ONE_MASK << (keyNum % 8);
      printf("Keynum HERE %i:\n", keyNum);      
   }
}
/*
// result ends up in y; x is also overwritten
__device__ void gcd(unsigned int *x, unsigned int *y) {
   int c = 0;

   if (isNonZero(x) && isNonZero(y)) {
      // __syncthreads(); // definitely needed here

      // we think this loop is okay
      while (((x[WORDS_PER_KEY - 1] | y[WORDS_PER_KEY - 1]) & 1) == 0) {
         shiftR1(x);
         shiftR1(y);
         c++;
      }

      while (__any(x[threadIdx.x])) {

         while ((x[WORDS_PER_KEY - 1] & 1) == 0) {
            shiftR1(x);
         }

         // SOMETHING BAD HAPPENS AROUND HERE

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

      // __syncthreads(); // definitely needed here

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
*/
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

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef LOW_ONE_MASK
#define LOW_ONE_MASK 0x01
#endif

#define NUM_STREAMS 1

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void mygcd(unsigned int *x, unsigned int *y);
__device__ void gcd(unsigned int *x, unsigned int *y);
__device__ void shiftR1(unsigned int *arr);
__device__ void shiftL1(unsigned int *arr);
__device__ void shiftL(unsigned int *arr, unsigned int x);
__device__ void subtract(uint32_t *x, uint32_t *y);
__device__ int geq(uint32_t *x, uint32_t *y);
__device__ int isNonZero(uint32_t *x);
__global__ void cuGCD(u1024bit_t *key, u1024bit_t *key_comparison_list, 
    uint8_t *bitvector);
__device__ int isGreaterThanOne(uint32_t *number);
static void HandleError( cudaError_t err, const char *file, int line);

extern "C"
{
   void dispatchGcdCalls(u1024bit_t *array, uint32_t *found, int count, FILE *dfp, FILE *nfp);
   void callCudaStreams(u1024bit_t *array, uint8_t *bitVector,
         u1024bit_t *d_keys, u1024bit_t *d_currentKey, uint8_t *d_bitVector,
         int j, int count, int stride, cudaStream_t stream);
}

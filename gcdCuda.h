#ifndef LOW_ONE_MASK
#define LOW_ONE_MASK 0x01
#endif

__device__ void gcd(unsigned int *x, unsigned int *y);
__device__ void shiftR1(unsigned int *arr);
__device__ void shiftL1(unsigned int *arr);
__device__ void shiftL(unsigned int *arr, unsigned int x);
__device__ void subtract(uint32_t *x, uint32_t *y);
__device__ int geq(uint32_t *x, uint32_t *y);
__device__ int isNonZero(uint32_t *x);
__global__ void cuGCD(u1024bit_t *key, u1024bit_t *key_comparison_list, 
    uint32_t *bitvector);
__device__ int isGreaterThanOne(uint32_t *number);

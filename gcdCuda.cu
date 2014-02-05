__device__ void shiftR1(unsigned int *arr)
{
   unsigned int index = threadIdx.x;
   uint32_t temp;

   if (index != 0)
   {
      temp = arr[index + 1];
   }
   else
   {
      temp = 0;
   }

   arr[i] >>= 1;
   arr[i] |= (temp << 31);
}

__device__ void shiftL1(unsigned int *arr)
{
   unsigned int index = threadIdx.x;
   uint32_t temp;

   if (index != 0)
   {
      temp = arr[index + 1];
   }
   else
   {
      temp = 0;
   }

   arr[i] <<= 1;
   arr[i] |= (temp >> 31);
}


/* x must be between 0 and 31  inclusive */
__device__ void shiftRX(unsigned int *arr, unsigned int x)
{
   unsigned int index = threadIdx.x;
   uint32_t temp;

   if (index != 0)
   {
      temp = arr[index + 1];
   }
   else
   {
      temp = 0;
   }

   arr[i] >>= x;
   arr[i] |= (temp << x);
}

/* x must be between 0 and 31  inclusive */
__device__ void shiftLX(unsigned int *arr, unsigned int x)
{
   unsigned int index = threadIdx.x;
   uint32_t temp;

   if (index != 0)
   {
      temp = arr[index + 1];
   }
   else
   {
      temp = 0;
   }

   arr[i] <<= x;
   arr[i] |= (temp >> x);
}

__device__ void subtract(uint32_t *x, uint32_t *y, uint32_t *result)
{
   __shared__ uint8_t borrow[WORDS_PER_KEY];

   uint8_t index = threadIdx.x;

   // initialize borrow array to 0
   borrow[index] = 0;
   __syncthreads();

   result[index] = x[index] - y[index];

   if (x[index] < y[index] && index > 0) {
      borrow[index - 1] = 1;
   }

   int underflow = 0;

   while (__any(borrow[index])) {
      if (borrow[index]) {
         underflow = result[index] < 1;
         result[index] = result[index] - 1;

         if (underflow && index > 0) {
            borrow[index - 1] = 1;
         }

         borrow[index] = 0;
      }
   }
}

__device__ int geq(uint32_t *x, uint32_t *y) {
   __shared__ uint8_t pos;

   uint8_t index = threadIdx.x;

   pos = WORDS_PER_KEY - 1;
   __syncthreads();

   if (x[index] != y[index]) {
      atomicMin(&pos, index);
   }

   return x[pos] >= y[pos];
}


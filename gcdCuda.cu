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

__device__ void subtract(unsigned int *x, unsigned int *y)
{
   unsigned int index = threadIdx.x;
   uint32_t temp = x[index];
   __shared__ unsigned int borrow[WORDS_PER_KEY];

   x[index] -= y[index];

   /* detect underflow */
   if (x[index] > temp)
   {
      borrow[index + 1] = 1;
   }
   
   // TODO
}

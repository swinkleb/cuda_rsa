#include "gmp_location.h"
#include "main.h"
#include "gcd.h"
#include "io.h"
#include "gcdCuda.h"

int main (int argc, char **argv)
{
   if (argc < MIN_ARG_COUNT)
   {   
      usage(argv[PROG_ARG]);
   }   

   switch (argv[FLAG_ARG][0])
   {   
      case 'c':
         cpuImpl(argv[IN_FILE_ARG], 
               argc > MIN_ARG_COUNT ? argv[D_OUT_FILE_ARG] : NULL,
               argc > MIN_ARG_COUNT + 1 ? argv[N_OUT_FILE_ARG + 1] : NULL);
         break;

      case 'g':
         gpuImpl(argv[IN_FILE_ARG],
               argc > MIN_ARG_COUNT ? argv[D_OUT_FILE_ARG] : NULL,
               argc > MIN_ARG_COUNT + 1 ? argv[N_OUT_FILE_ARG + 1] : NULL);
         break;

      default:
         usage(argv[PROG_ARG]);
   }   
}

void usage(char *myName)
{
   printf("Usage: %s <flag> <input_file> [d_output_file] [n_output_file]\n", myName);
   printf("Flags:\n");
   printf("\tc - cpu implementation\n");
   printf("\tg - gpu implementation\n");

   exit(1);
}

void cpuImpl(char *inFile, char *dOutFile, char *nOutFile)
{
   mpz_t *array;
   unsigned int count;

   count = readKeysFromFileMPZ(&array, inFile);
   count = findGCDs(array, count, 
         dOutFile == NULL ? DEFAULT_D_OUT_FILE : dOutFile,
         nOutFile == NULL ? DEFAULT_N_OUT_FILE : nOutFile
         );
   
   printf("Total number of bad keys found: %d\n", count);
}

void gpuImpl(char *inFile, char *dOutFile, char *nOutFile)
{
   u1024bit_t *array;
   uint32_t *found;
   unsigned int count;

   count = readKeysFromFile(&array, inFile);

   found = (uint32_t *) calloc(ceil(((float) count / (float) WORD_SIZE)), sizeof(uint32_t));
   if (NULL == found)
   {
      perror("calloc");
      exit(1);
   }

   dispatchGcdCalls(array, found, count, dOutFile == NULL ? DEFAULT_D_OUT_FILE : dOutFile);
}

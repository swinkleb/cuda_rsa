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

   /* keeps track of which keys have already been outputted */
   found = (uint32_t *) calloc(ceil(((float) count / (float) WORD_SIZE)), sizeof(uint32_t));
   if (NULL == found)
   {
      perror("calloc");
      exit(1);
   }

   /* open output files so we can pass FILE *'s */
   FILE *dfp = fopen(dOutFile == NULL ? DEFAULT_D_OUT_FILE : dOutFile, "w");
   if (NULL == dfp) 
   {   
      perror("opening file");
      exit(1);
   } 

   FILE *nfp = fopen(nOutFile == NULL ? DEFAULT_N_OUT_FILE : nOutFile, "w");
   if (NULL == nfp) 
   {   
      perror("opening file");
      exit(1);
   } 

   dispatchGcdCalls(array, found, count, dfp, nfp);

   /* close output files */
   fclose(dfp);
   fclose(nfp);
}

// can use this for debugging
// prints hex and decimal values for u1024bit_t's number array
void print1024Int(uint32_t *number) {
   static int printedNum = 0;

   printf("%i HEX:\n", printedNum);

   int i;
   for (i = 0; i < NUM_INTS; i++) {
      printf("%20x ", number[i]);

      if ((i + 1) % 4 == 0 && i > 0) {
         printf("\n");
      }
   }

   printf("\n%i DEC:\n", printedNum++);

   for (i = 0; i < NUM_INTS; i++) {
       printf("%20u ", number[i]);

      if ((i + 1) % 4 == 0 && i > 0) {
         printf("\n");
      }
   }
   printf("\b");
   printf("\n\n");
}

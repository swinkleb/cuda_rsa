#include "gcd.h"
#include "main.h"
#include "io.h"

int main (int argc, char **argv)
{
   if (argc < MIN_ARG_COUNT)
   {   
      usage(argv[PROG_ARG]);
   }   

   switch (argv[FLAG_ARG][0])
   {   
      case 'c':
         cpuImpl(argv[IN_FILE_ARG], argc > MIN_ARG_COUNT ? argv[OUT_FILE_ARG] : NULL);
         break;

      case 'g':
         gpuImpl(argv[IN_FILE_ARG], argc > MIN_ARG_COUNT ? argv[OUT_FILE_ARG] : NULL);
         break;

      default:
         usage(argv[PROG_ARG]);
   }   
}

void usage(char *this)
{
   printf("Usage: %s <flag> <input_file> [output_file]\n", this);
   printf("Flags:\n");
   printf("\tc - cpu implementation\n");
   printf("\tg - gpu implementation\n");

   exit(1);
}

void cpuImpl(char *inFile, char *outFile)
{
   mpz_t *array;
   unsigned int count;

   count = readKeysFromFileMPZ(&array, inFile);
   count = findGCDs(array, count, outFile == NULL ? DEFAULT_OUT_FILE : outFile);
   
   printf("Total number of bad keys found: %d\n", count);
}

void gpuImpl(char *inFile, char *outFile)
{
   u1024bit_t *array;
//   u1024bit_t *privateKeys;
   uint32_t *found;
   unsigned int count;

   count = readKeysFromFile(&array, inFile);
   
   found = (uint32_t *) calloc(ceil(((float) count / (float) WORD_SIZE)), sizeof(uint32_t));
   if (NULL == found)
   {   
      perror("calloc");
      exit(1);
   }   

//   count = findGCDs(array, count, &privateKeys);
   outputKeysToFile(privateKeys, count, outFile == NULL ? DEFAULT_OUT_FILE : outFile);
}

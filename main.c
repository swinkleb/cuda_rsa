#include "io.h"
#include "gcd.h"
#include "main.h"

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
   mpz_t *privateKeys;
   unsigned int count;

   count = readKeysFromFileMPZ(&array, inFile);
   count = findGCDs(array, count, &privateKeys);
   outputKeysToFileMPZ(privateKeys, count, outFile == NULL ? DEFAULT_OUT_FILE : outFile);
}

void gpuImpl(char *inFile, char *outFile)
{
/* TODO
   uint1024 *array;
   uint1024 *privateKeys;
   unsigned int count;

   count = readKeysFromFile(&array, inFile);
   count = findGCDs(array, count, &privateKeys);
   outputKeysToFile(privateKeys, count, outFile == NULL ? DEFAULT_OUT_FILE : outFile);
*/
}

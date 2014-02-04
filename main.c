#include "io.h"
#include "gcd.h"
#include "main.h"

int main (int argc, char **argv)
{
   if (argc < ARG_COUNT)
   {   
      usage(argv[PROG_ARG]);
   }   

   switch (argv[FLAG_ARG][0])
   {   
      case 'c':
         cpuImpl(argv[FILE_ARG]);
         break;

      case 'g':
         gpuImpl(argv[FILE_ARG]);
         break;

      default:
         usage(argv[PROG_ARG]);
   }   
}

void usage(char *this)
{
   printf("Usage: %s <flag> <input_file>\n", this);
   printf("Flags:\n");
   printf("\tc - cpu implementation\n");
   printf("\tg - gpu implementation\n");

   exit(1);
}

void cpuImpl(char *filename)
{
   mpz_t *array;
   mpz_t *priv;
   unsigned int count;

   count = readKeysFromFileMPZ(&array, filename);
   count = findGCDs(array, count, &priv);
   outputKeysToFileMPZ(priv, count, "output.txt");
}

void gpuImpl(char *filename)
{
   //TODO
}

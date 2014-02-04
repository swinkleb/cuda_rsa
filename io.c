#include "io.h"

int readKeysFromFile(uint1024 **keys, char *filename)
{
   mpz_t temp;
   size_t wordCount;
   unsigned int count;
   unsigned int curArraySize = DEFAULT_SIZE;
   FILE *fp = fopen(filename, "r");
   if (NULL == fp)
   {
      perror("opening file");
      exit(1);
   }

   mpz_init(temp);

   *keys = (uint1024 *) malloc(curArraySize * sizeof(uint1024));
   if (NULL == *keys)
   {
      perror("malloc");
      exit(1);
   }

   /* read in each line of file as a key */
   for (count = 0; mpz_inp_str(temp, fp, BASE_10); count++)
   {
      /* ran out of space need to make the array bigger */
      if (count >= curArraySize)
      {
         curArraySize += DEFAULT_SIZE;
         *keys = realloc(*keys, sizeof(uint1024) * curArraySize);
         if (NULL == *keys)
         {
            perror("realloc");
            exit(1);
         }
      }

      /* store in uint1024 so least significant word is first */
      mpz_export(&((*keys)[count].words[0]), &wordCount, -1, BYTES_IN_WORD, 0, 0, temp);
      /* if value does not have the full amount of words expected pad with words value of 0 */
      if (wordCount < WORDS_PER_KEY) //TODO check if faster to memset just memset everything everytime
      {
         for (int i = wordCount; i < WORDS_PER_KEY; i++)
         {
            (*keys)[count].words[i] = 0;
         }
      }

      /* set all bits to 0 so temp is ready for the next iter */
      mpz_set_ui(temp, 0);
   }

   fclose(fp);
   mpz_clear(temp);
   
   return count;
}

void outputKeysToFile(uint1024 *keys, unsigned int count, char *filename)
{
   mpz_t temp;
   FILE *fp = fopen(filename, "w");
   if (NULL == fp)
   {
      perror("opening file");
      exit(1);
   }

   mpz_init(temp);

   for (int i = 0; i < count; i++)
   {
      mpz_import(temp, WORDS_PER_KEY, -1, BYTES_IN_WORD, 0, 0, &(keys[i].words[0]));
      mpz_out_str(fp, BASE_10, temp);
      fprintf(fp, "\n");

      /* set all bits to 0 so temp is ready for the next iter */
      mpz_set_ui(temp, 0);
   }

   fclose(fp);
   mpz_clear(temp);
}
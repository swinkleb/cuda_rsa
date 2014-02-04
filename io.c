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
      for (int i = wordCount; i < WORDS_PER_KEY; i++)
      {
         (*keys)[count].words[i] = 0;
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


int readKeysFromFileMPZ(mpz_t **keys, char *filename)
{
   unsigned int count = 0;
   unsigned int curArraySize = DEFAULT_SIZE;
   FILE *fp = fopen(filename, "r");
   if (NULL == fp)
   {
      perror("opening file");
      exit(1);
   }

   *keys = (mpz_t *) malloc(1 + curArraySize * sizeof(mpz_t));
   if (NULL == *keys)
   {
      perror("malloc");
      exit(1);
   }

   mpz_init(*keys[count]);
   /* read in each line of file as a key */
   for (; mpz_inp_str((*keys)[count], fp, BASE_10); count++)
   {
      /* ran out of space need to make the array bigger */
      if (count + 1 >= curArraySize)
      {
         curArraySize += DEFAULT_SIZE;
         *keys = realloc(*keys, curArraySize * sizeof(mpz_t));
         if (NULL == *keys)
         {
            perror("realloc");
            exit(1);
         }
      }
      
      if (count + 1 < curArraySize)
      {
         mpz_init((*keys)[count + 1]);
      }
   }

   fclose(fp);
   
   return count;
}

void outputKeysToFileMPZ(mpz_t *keys, unsigned int count, char *filename)
{
   FILE *fp = fopen(filename, "w");
   if (NULL == fp)
   {
      perror("opening file");
      exit(1);
   }


   for (int i = 0; i < count; i++)
   {
      mpz_out_str(fp, BASE_10, keys[i]);
      fprintf(fp, "\n");
   }

   fclose(fp);
}

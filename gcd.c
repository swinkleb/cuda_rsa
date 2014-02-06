#include "gcd.h"
#include "rsa.h"

int findGCDs(mpz_t *arr, unsigned int size, char *filename)
{
   unsigned int count = 0;
   uint32_t *found;
   mpz_t p;
   mpz_t q;
   mpz_t d;

   mpz_inits(p, q, d, NULL);
   found = (uint32_t *) calloc(ceil(((float) size / (float) WORD_SIZE)), sizeof(uint32_t));
   if (NULL == found)
   {
      perror("calloc");
      exit(1);
   }

   FILE *fp = fopen(filename, "w");
   if (NULL == fp) 
   {   
      perror("opening file");
      exit(1);
   }   
  
   for (int i = 0; i < size; i++)
   {
      for (int j = i + 1; j < size; j++)
      {
         mpz_gcd(p, arr[i], arr[j]);
        
         /* found a common factor */
         if (mpz_cmp_ui(p, (unsigned long) 1) > 0)
         {
            /* check if previously found */
            if (!isFound(found, i))
            {
               mpz_cdiv_q(q, arr[i], p);
               calcPrivateKey(p, q, &d);
               mpz_out_str(fp, BASE_10, d);
               fprintf(fp, "\n");
             
               setFound(found, i);
               count++;
               
               /* debug code */
/*
               mpz_out_str(stdout, BASE_10, arr[i]);
               printf("\n");
*/
            }

            /* check if previously found */
            if (!isFound(found, j))
            {
               mpz_cdiv_q(q, arr[j], p);
               calcPrivateKey(p, q, &d);
               mpz_out_str(fp, BASE_10, d);
               fprintf(fp, "\n");
               
               setFound(found, j);
               count++;
                  
               /* debug code */
/*
               mpz_out_str(stdout, BASE_10, arr[j]);
               printf("\n");
*/
            }
         }
      }
      fflush(fp);
   }
   
   mpz_clears(p, q, d, NULL);
   free(found);
   fclose(fp);

   return count;
}

void setFound(uint32_t *arr, int bit)
{
   int index = bit / WORD_SIZE;
   int offset = bit % WORD_SIZE;

   arr[index] |= 1 << offset;
}

int isFound(uint32_t *arr, int bit)
{
   int index = bit / WORD_SIZE;
   int offset = bit % WORD_SIZE;

   return arr[index] & (1 << offset);
}

int gcd(mpz_t r, mpz_t a, mpz_t b)
{
   mpz_t temp;
   mpz_init(temp);

   /* if (b > a) */
   if (mpz_cmp(a, b) < 0)
   {   
      mpz_set(temp, a);
      mpz_set(a, b);
      mpz_set(b, temp);
   }   

   /* while (b != 0) */
   while (mpz_cmp_ui(b, (unsigned int) 0) != 0) 
   {   
      mpz_set(temp,b);
      mpz_mod(b, a, b);
      mpz_set(a,  temp);
   }

   mpz_set(r, a);

   return mpz_cmp_ui(a, (unsigned int) 1) != 0;
}

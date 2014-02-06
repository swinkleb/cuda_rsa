#include "rsa.h"
#include "main.h"
#include "io.h"
#include "gcd.h"

int findGCDs(mpz_t *arr, unsigned int size, const char *filename)
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

int computeAndOutputGCDs(u1024bit_t *arr, uint32_t *found, uint8_t *bitvector, int commonKeyOffset, int iOffset, const char *filename)
{
   unsigned int thisKeyOffset;
   unsigned int count = 0;
   mpz_t p;
   mpz_t q;
   mpz_t d;
   mpz_t temp1;
   mpz_t temp2;

   mpz_inits(p, q, d, temp1, temp2, NULL);

   /* move this outside function call */
   FILE *fp = fopen(filename, "w");
   if (NULL == fp) 
   {   
      perror("opening file");
      exit(1);
   }   
   /* end */

   /* each bitvector */
   for (int i = 0; i < NUM_BLOCKS; i++)
   {
      /* each bit per bitvector */
      for (int j = 0; j < BLOCK_DIM_Y; j++)
      {
         /* if corresponding bit is set in bit vector, found a common factor */
         if (bitvector[i] & (1 << j))
         {
            thisKeyOffset = iOffset + i * BLOCK_DIM_Y + j;
            
            mpz_import(temp1, WORDS_PER_KEY, 1, BYTES_IN_WORD, 0, 0, 
                  &(arr[commonKeyOffset].number[0]));

            mpz_import(temp2, WORDS_PER_KEY, 1, BYTES_IN_WORD, 0, 0, 
                  &(arr[thisKeyOffset].number[0]));

            mpz_gcd(p, temp1, temp2);

            /* check if previously found */
            if (!isFound(found, commonKeyOffset))
            {
               mpz_cdiv_q(q, temp1, p);
               calcPrivateKey(p, q, &d);
               mpz_out_str(fp, BASE_10, d);
               fprintf(fp, "\n");

               setFound(found, commonKeyOffset);
               count++;

               /* debug code */
/*
                  mpz_out_str(stdout, BASE_10, temp1);
                  printf("\n");
*/
            }

            /* check if previously found */
            if (!isFound(found, thisKeyOffset))
            {
               mpz_cdiv_q(q, temp2, p);
               calcPrivateKey(p, q, &d);
               mpz_out_str(fp, BASE_10, d);
               fprintf(fp, "\n");

               setFound(found, thisKeyOffset);
               count++;

               /* debug code */
/*
                  mpz_out_str(stdout, BASE_10, temp2);
                  printf("\n");
*/
            }
         }
      }
      fflush(fp);
   }
   
   mpz_clears(p, q, d, temp1, temp2, NULL);
   /* Move this ouside call*/
   fclose(fp);
   /* end */

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

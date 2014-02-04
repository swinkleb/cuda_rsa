#include "gcd.h"
#include "rsa.h"

int findGCDs(mpz_t *arr, unsigned int size, mpz_t **privates)
{
   unsigned int count = 0;
   unsigned int curArraySize = EXPECTED_KEY_NUM;
   mpz_t p;
   mpz_t q;
   mpz_t d;

   mpz_inits(p, q, d, NULL);

   *privates = (mpz_t *) malloc(curArraySize * sizeof(mpz_t));
   if (NULL == *privates)
   {
      perror("malloc");
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
            mpz_cdiv_q(q, arr[i], p);
            calcPrivateKey(p, q, &d);
            mpz_init((*privates)[count]);
            mpz_set((*privates)[count], d);
            count++;

            mpz_cdiv_q(q, arr[j], p);
            calcPrivateKey(p, q, &d);
            mpz_init((*privates)[count]);
            mpz_set((*privates)[count], d);
            count++;
         }
         
         /* ran out of space need to make the array bigger */
         if (count >= curArraySize)
         {
            curArraySize += EXPECTED_KEY_NUM;
            *privates = realloc(*privates, sizeof(mpz_t) * curArraySize);
            if (*privates == NULL)
            {
               perror("realloc");
               exit(1);
            }
         }
      }
   }

   mpz_clears(p, q, d, NULL);

   return count;
}

#include "gmp_location.h"
#include "rsa.h"

void calcPrivateKey(mpz_t p, mpz_t q, mpz_t *d)
{
   mpz_t temp1;
   mpz_t temp2;
   mpz_t e;
   mpz_t n;
   mpz_t tot;

   mpz_init_set_ui(e, RSA_E);
   mpz_inits(n, *d, temp1, temp2, tot, NULL);

   /* calc n = pq */
   mpz_mul(n, p, q);

   /* calc tot(n) = (p - 1)(q - 1) */
   mpz_sub_ui(temp1, p, (unsigned long) 1);
   mpz_sub_ui(temp2, q, (unsigned long) 1);
   mpz_mul(tot, temp1, temp2);

   /* calc d = 1/e mod tot(n) */
   mpz_invert(*d, e, tot);

   mpz_clears(e, n, temp1, temp2, tot, NULL);
}

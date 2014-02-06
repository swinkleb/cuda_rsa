#include <stdio.h>
#include "/home/clupo/gmp/include/gmp.h"
//#include <gmp.h>

#define RSA_E 65537

void calcPrivateKey(mpz_t p, mpz_t q, mpz_t *d);

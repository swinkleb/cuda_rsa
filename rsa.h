/*
 * Scott Winkleblack
 * Assignment 1
 * csc 556
 */

#include <stdio.h>
#include <gmp.h>

/*
 * GCD pseudo code
 *
 * if x and y are even then
 *    return 2 * GCD(x/2, y/x)
 * else if x is even and y is odd then
 *    return GCD(x/2, y)
 * else if x is odd and y is even
 *    return GCD(y/2, x)
 * else 
 *    if x >= y
 *       return GCD((x-y)/2, y)
 *    else
 *       return GCD (y-x)/2, x)
 *
 */

/* 
 * Parallel right shift pseudo code
 *
 * if treadID != 0
 *    temp = x[threadID - 1]
 * else 
 *    temp = 0
 *
 * x = x >> 1
 * x = x | (temp << 31)
 *
 */

typedef struct {
   mpz_t p;
   mpz_t q;
   mpz_t n;  
   mpz_t e; /* public */
   mpz_t d; /* private */
} rsa;
void genRandPrime(mpz_t rand);

void outputVals(char *file, mpz_t *arr, int size);

void readVals(FILE *in, mpz_t **arr, int *size);

void pairwiseGCD(mpz_t *arr, mpz_t *gcd, int size, FILE *out);

void generateKeys(rsa *key);

void rsaEncrypt(mpz_t m, mpz_t c, rsa *key);

void rsaDecrypt(mpz_t m, mpz_t c, rsa *key);

void usage(char *name);

void textbookRSA(FILE *in, FILE *out);

void findAndOutputCommonFactors(FILE *in, FILE *out);

void fillInKey(mpz_t p, mpz_t q, rsa *key);

void decryptMsg(FILE *moduli, FILE *ciphertext);



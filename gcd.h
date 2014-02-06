#include <stdlib.h>
#include <math.h>
#include <stdio.h>
//#include "/home/clupo/gmp/include/gmp.h"
#include <gmp.h>

#define EXPECTED_KEY_NUM 1000
#define WORD_SIZE 32
#define BASE_10 10

int findGCDs(mpz_t *arr, unsigned int size, char *filename);

void setFound(uint32_t *arr, int bit);

int isFound(uint32_t *arr, int bit);

int gcd(mpz_t r, mpz_t a, mpz_t b);

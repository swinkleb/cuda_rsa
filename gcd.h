#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define EXPECTED_KEY_NUM 1000
#define WORD_SIZE 32
#define BASE_10 10

int findGCDs(mpz_t *arr, unsigned int size, const char *dfilename, const char *nfilename);

int computeAndOutputGCDs(u1024bit_t *arr, uint32_t *found, uint8_t *bitvector, int commonKeyOffset, int iOffset, FILE *dfp, FILE *nfp);

void setFound(uint32_t *arr, int bit);

int isFound(uint32_t *arr, int bit);

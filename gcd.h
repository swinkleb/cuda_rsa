#include <stdlib.h>
#include <math.h>
#include <stdio.h>
//#include "/home/clupo/gmp/include/gmp.h"
#include <gmp.h>

#define EXPECTED_KEY_NUM 1000
#define WORD_SIZE 32
#define BASE_10 10

int findGCDs(mpz_t *arr, unsigned int size, char *filename);

int computeAndOutputGCDs(u1024bit_t *arr, uint32_t *found, uint8_t *bitvector, int commonKeyOffset, int iOffset, char *filename);

void setFound(uint32_t *arr, int bit);

int isFound(uint32_t *arr, int bit);

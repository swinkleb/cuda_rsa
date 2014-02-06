#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "/home/clupo/gmp/include/gmp.h"
//#include <gmp.h>

#define BASE_10 10
#define WORD_SIZE 32
#define BYTES_IN_WORD (WORD_SIZE / 8)
#define KEY_SIZE 1024
#define WORDS_PER_KEY (KEY_SIZE / WORD_SIZE) 
#define DEFAULT_SIZE 200000

int readKeysFromFile(u1024bit_t **keys, char *filename);

void outputKeysToFile(u1024bit_t *keys, unsigned int count, char *filename);

int readKeysFromFileMPZ(mpz_t **keys, char *filename);

void outputKeysToFileMPZ(mpz_t *keys, unsigned int count, char *filename);

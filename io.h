#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <gmp.h>

#define BASE_10 10
#define WORD_SIZE 32
#define BYTES_IN_WORD (WORD_SIZE / 8)
#define KEY_SIZE 1024
#define WORDS_PER_KEY (KEY_SIZE / WORD_SIZE) 
#define DEFAULT_SIZE 200000

typedef struct {
      uint32_t words[WORDS_PER_KEY];
} uint1024;

int readKeysFromFile(uint1024 **keys, char *filename);

void outputKeysToFile(uint1024 *keys, unsigned int count, char *filename);

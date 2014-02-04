#include "rsa.h"
#include "io.h"

int main (int argc, char **argv)
{
   uint1024 *array;
   int count = readKeysFromFile(&array, "input.txt");
   outputKeysToFile(array, count, "output.txt");
}

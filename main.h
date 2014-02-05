#define MIN_ARG_COUNT 3
#define PROG_ARG 0
#define FLAG_ARG 1
#define IN_FILE_ARG 2
#define OUT_FILE_ARG 3
#define DEFAULT_OUT_FILE "output.txt"

void usage(char *this);

void cpuImpl(char *inFile, char *outFile);

void gpuImpl(char *inFile, char *outFIle);

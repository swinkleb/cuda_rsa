#!/bin/bash

#export PATH=$PATH:/usr/local/cuda-5.5/bin/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/clupo/gmp/lib/

clear
echo "================ Start ==============="
echo "Input: input/$1-keys.txt"
echo ""

echo "GPU Time:"
time ./rsa g "input/$1-keys.txt" "output/gpu-$1-bad.txt"
echo ""
echo "CPU Time:"
time ./rsa c "input/$1-keys.txt" "output/cpu-$1-bad.txt"

echo "================= diffing D files ================="
diff "output/gpu-$1-bad.txt" "output/gpu-$1-bad.txt"

echo "================= Fin ================="

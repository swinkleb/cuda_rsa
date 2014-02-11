import sys, random

if len(sys.argv) < 3:
	print "usage: python ndFormat.py (n input file) (d input file) [output file name]"
	exit()

if len(sys.argv) >= 4:
	outputName = sys.argv[3]
else:
	outputName = "output.txt"

nFile = open(sys.argv[1], 'r')
dFile = open(sys.argv[2], 'r')

nLines = nFile.readlines()
nFile.close()

dLines = dFile.readlines()
dFile.close()

outputFile = open(outputName, 'w')

for i in xrange(len(nLines)):
	outputFile.write(nLines[i].rstrip() + ":" + dLines[i].rstrip() + "\n")

outputFile.close()

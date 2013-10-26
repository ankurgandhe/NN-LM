import sys

def printFeatures(fprompts):
	for l in open(fprompts):
		id = int(l.strip().split()[0])
		n = len(l.strip().split())
		for i in range(n):
			print id,id,id

fprompts = sys.argv[1]
printFeatures(fprompts)

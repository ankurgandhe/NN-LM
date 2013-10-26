import sys 
from Corpus import ReadVocabFile 

def ReadMissingUnigram(fmissingbigram,WordID):
	foundBigrams = {}
	for l in open(fmissingunigrams):
		l=l.strip().split('\t')
		w1 = l[0].strip()
		if w1.find('<s>')>=0:
			continue 
		foundBigrams[WordID[w1]] = 1
	print >> sys.stderr, len(foundBigrams)
	return foundBigrams 


fvocabid, fmissingunigrams , fbigrams = sys.argv[1:] 


WordID,printMap  = ReadVocabFile(fvocabid)
missingBigrams = ReadMissingUnigram(fmissingunigrams,WordID)

for l in open(fbigrams):
	l=l.strip().split()
	wid1 = int(l[0].strip())
	wid2 = int(l[1].strip())
	if (wid2) in missingBigrams:
		print 1
	else:
		print 0 

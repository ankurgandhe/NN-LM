import sys 
from Corpus import ReadVocabFile 

def ReadMissingBigram(fmissingbigram,WordID):
	foundBigrams = {}
	for l in open(fmissingbigrams):
		l=l.strip().split()
		w1 = l[0].strip()
		w2 = l[1].strip()
		if w1.find('<s>')>=0 or w2.find('</s>')>=0:
			continue 
		foundBigrams[(WordID[w1],WordID[w2])] = 1
	print >> sys.stderr, len(foundBigrams)
	return foundBigrams 


fvocabid, fmissingbigrams , fbigrams = sys.argv[1:] 


WordID,printMap  = ReadVocabFile(fvocabid)
missingBigrams = ReadMissingBigram(fmissingbigrams,WordID)

for l in open(fbigrams):
	l=l.strip().split()
	wid1 = int(l[0].strip())
	wid2 = int(l[1].strip())
	if (wid1,wid2) in missingBigrams:
		print 1
	else:
		print 0 

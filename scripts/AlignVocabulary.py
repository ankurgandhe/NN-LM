import sys
from Corpus import ReadVocabFile, GetSynset
''' 
'' If the vocabularies have same words, or there is a dictionary present, align the vobucalaries so that there is one to
'' one mapping in topN words. 
''' 

def PrintAlignVocab(WordID1, WordID2, synset,fout1,fout2):
    Markers = ['<UNK>','<s>','</s>']    
    counter = 0 
    for  w in synset:
	if w in Markers:
	    continue 
	print >> fout1, w
	print >> fout2, w 
        counter = counter + 1 
    print >> sys.stderr, "Number of same words:", counter 
    for w in WordID1:
	if w not in synset:
	    print >> fout1, w
    for w in WordID2:
	if w not in synset:
	    print >> fout2, w

def GetSyn(WordID1, WordID2):
    synset = []
    for w in WordID1:
	if w in WordID2:
	    synset.append(w)
    return synset 

if __name__ == '__main__':
    if len(sys.argv)<2:
        print >> sys.stderr, " usage : python AlignVocab.py vocab1 vocab2  "
        sys.exit(0)

    vocab1 , vocab2 = sys.argv[1:]
    WordID1,a = ReadVocabFile(vocab1)
    WordID2,a = ReadVocabFile(vocab2)	
    synset = GetSyn(WordID1,WordID2)
    out1 = vocab1+".aligned"
    out2 = vocab2+".aligned"
    print >> sys.stderr, "Writing vocabs into files:",out1,",",out2
    fout1 = open(out1,"w")
    fout2  = open(out2,"w")
    PrintAlignVocab(WordID1, WordID2, synset,fout1,fout2)
	




import sys
from Corpus import ReadVocabFile

def ReadFreqFile(fp):
    wF={}
    for l in open(fp):
        l=l.strip().split()
        if len(l)<2:
            print >> sys.stderr, "Voc frequency entry \'",l,"\'does not have either word or frequency... not using freq."
            return []
            break
        w = l[1]
        fr = int(l[0])
        #if fr > 1:
        #    break
        #UNKw.append(w)
	wF[w] = fr 
    wF["<UNK>"]=0
    wF["<s>"]=0
    wF["</s>"]=0
    return wF

fvocab, ffreq = sys.argv[1:]
Vocab,pr = ReadVocabFile(fvocab)
Freq = ReadFreqFile(ffreq)
fr=0
totfr=0
for w in Vocab:
	if w.find("<")>=0:
		continue 
	if Vocab[w]>603 or Vocab[w]<=2:
		totfr = totfr + Freq[w]
		continue 
	print w,Vocab[w],Freq[w]
	fr = fr + Freq[w]
	totfr = totfr + Freq[w]

print fr , totfr , float(fr*100)/totfr
	

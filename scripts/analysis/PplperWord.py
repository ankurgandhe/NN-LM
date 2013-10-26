import sys 
import math 
def GetPplPerWord(fppl):
    Words = {}
    Count = {} 
    for l in open(fppl):
	l=l.strip()
	if l.find("p(")>=0:
	    l=l.strip().split()
	    w = l[1]
	    lprob = float(l[-2])
	    if w not in Words:
		Words[w] = 0.0 
	    	Count[w] = 0
	    Words[w] = Words[w] + lprob 
	    Count[w] = Count[w] + 1 	
    for w in Words:
	Words[w] = Words[w] / Count[w]
    return Words,Count

def ReadFreqFile(fp):
    Freq={}
    for l in open(fp):
        l=l.strip().split()
        if len(l)<2:
            print >> sys.stderr, "Voc frequency entry \'",l,"\'does not have either word or frequency... not using freq."
            return []
            break
        w = l[1]
        fr = int(l[0])
	Freq[w] = fr 
    return Freq 

def GetPplPerFreq(Words,ffreq):
    Freq = ReadFreqFile(ffreq)
    Bins = {}
    Counts = {}
    for w in Words:
	if w in Freq:
	    fr = int(Freq[w])
	else:
	    fr = 0
	if fr not in Bins:
	    Bins[fr] = 0.0
	    Counts[fr] = 0
	Bins[fr]  = Bins[fr]+Words[w]
	Counts[fr] = Counts[fr]+1
    for fr in Bins:
	Bins[fr] = Bins[fr] / Counts[fr]
    return Bins,Counts

def ReadKwList(fkwlist):
    KW = [] 
    for l in open(fkwlist):
	l=l.strip()
	if l.find("kwtext")>=0:
	    kw = l.split('>')[1].split('<')[0].strip()
	    for w in kw.split():
		KW.append(w)
    return KW 
def GetPplPerKW(Words,fkwlist):
    KW = ReadKwList(fkwlist)
    Bins = {}
    Counts = {}
    for w in Words:
	if w in KW:
	    if w not in Bins:
            	Bins[w] = 0.0
            	Counts[w] = 0
	    
	    Bins[w]  = Bins[w]+Words[w]
            Counts[w] = Counts[w]+1
	
    for w in Bins:
        Bins[w] = Bins[w] / Counts[w]
    return Bins,Counts


if len(sys.argv)<2:
    print "need 1/2/3 inputs : ppl-log-file \n ppl-log-file word-freq-file \n ppl-log-file kwlist-file(xml) kw"
    sys.exit(0)
fppl = sys.argv[1]
if len(sys.argv)>3:
    fkwlist = sys.argv[2]
    words,counts = GetPplPerWord(fppl)
    kws,counts = GetPplPerKW(words, fkwlist)
    for kw in kws:
        print kw,kws[kw]#,counts[kw]

elif len(sys.argv)>2: 
    ffreq = sys.argv[2]
    words,counts = GetPplPerWord(fppl)
    Freqs,counts = GetPplPerFreq(words, ffreq) 
    for fr in Freqs:
	print fr,Freqs[fr]#,counts[fr]
else:
    words,counts = GetPplPerWord(fppl)
    for w in words:
	print w, words[w],counts[w]


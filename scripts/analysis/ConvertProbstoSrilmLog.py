import sys 
import math 
def writeLog(fprob,ftextfile,ngram):
    fp = open(fprob,'r')
    totwords = 0
    totprob = 0.0
    totoov = 0
    print "Writing ARPA format Log for NNLM"
    print "ngram=",ngram 
    for l in open(ftextfile):
        l=l.strip()
        sentwords = 0
        sentprob = 0.0
        sentoov = 0
        print l

        history = ["<s>","<s>","<s>","<s>"]
        for w in l.split():
	    hLine = "" 
            for hw in history[len(history)-ngram+1:]:
                hLine = hLine + hw+" "
            nword = w 
            prob = float(fp.readline().strip())
            if prob == 0:
                nword = "<UNK>"
		lprob = 0.0	
	    else:
            	lprob = math.log(prob,10)
            print "\tp(",nword,"|",hLine,") =","["+str(ngram)+"gram]", prob,"[",lprob,"]" 
            if prob==0:
                sentoov = sentoov + 1
            else:
                sentwords = sentwords + 1 
                sentprob = sentprob + lprob 
	    i=0
            for ch in history[0:len(history)-1]:
                history[i]=history[i+1]
                i=i+1
            history[i]=w
	hLine = ""
	for hw in history[len(history)-ngram+1:]:
                hLine = hLine + hw+" "
        nword = "</s>"
        prob = float(fp.readline().strip())
	lprob = math.log(prob,10)
        print "\tp(",nword,"|",hLine,") =","["+str(ngram)+"gram]", prob,"[",lprob,"]"
	sentwords = sentwords + 1
	sentprob = sentprob + lprob

        totprob = totprob + sentprob
        totwords = totwords + sentwords 
        totoov = totoov + sentoov 

        print "1 sentences,",sentwords,"words,","- OOVs"
        print "0 zeroprobs,","logprob=",sentprob,"ppl=",math.pow(10, -1*(sentprob/sentwords)),"\n"
    print "file",ftextfile,":",str(totwords+totoov),"words,",totoov,"oovs"
    print "0 zeroprobs, logprob=",totprob,"ppl=",math.pow(10,-1*(totprob/totwords))

if len(sys.argv)<3:
    print "need 3 inputs : probility-file dev-text-file ngram"
    sys.exit(0)
fprob,ftextfile,ngram = sys.argv[1:]
writeLog(fprob,ftextfile,int(ngram))


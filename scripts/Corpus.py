import sys , math 
history = ["<s>","<s>","<s>","<s>"]

def GetVocabAndUNK(fvocab,fvocabfreq,ngram,add_unk,use_unk):
    UNKw=[]
    if (fvocabfreq.strip()==""):
        print >> sys.stderr, "frequency file not given"
        add_unk=False
        use_unk=False
    else:
        UNKw = ReadFreqFile(fvocabfreq)
        if UNKw==[]:
            add_unk=False
            use_unk=False
    (WordID,printMapFile) = ReadVocabFile(fvocab,UNKw,use_unk)
    '''if printMapFile:
        fwrite = open(fvocab+".nnid",'w')
        for w in sorted(WordID, key=WordID.get):
            print >> fwrite, w,WordID[w]
    '''
    return WordID,UNKw,printMapFile
     
def CreateData(ftrain,WordID,UNKw,ngram,add_unk,use_unk):
    if add_unk:
        TrainingData = PrepareData_UNK(ftrain,ngram,WordID,UNKw)
    elif use_unk:
        TrainingData = PrepareData(ftrain,ngram,WordID)
    else:
        TrainingData = PrepareData(ftrain,ngram,WordID)

    N_Vocab = max(WordID.itervalues())+1
    if use_unk == True:
        N_UNKw  = len(UNKw)
    else:
        N_UNKw = 0
    return TrainingData,N_Vocab,N_UNKw

def ReadVocabFile(fp,UNKw,use_unk):
    WordID={}
    WordID['<s>']=0
    WordID['</s>']=1
    WordID['<UNK>']=2
    idx=3
    printMap = True
    for l in open(fp):
        l=l.strip().split()
        if len(l)==2 and idx>0:
            print >> sys.stderr, "Vocab file with word ids given... using given word ids."
            idx=-1
            WordID={}
            printMap = False
        if idx>0:
            w=l[0].strip()
	    if w in WordID:
		continue 
            WordID[w]=idx
            idx=idx+1
        else:
            w = l[0].strip()
            wid = int(l[1])
            WordID[w]=wid
    return WordID,printMap



def ReadFreqFile(fp):
    UNKw=[]
    for l in open(fp):
        l=l.strip().split()
        if len(l)<2:
            print >> sys.stderr, "Voc frequency entry \'",l,"\'does not have either word or frequency... not using freq."
            return []
            break
        w = l[1]
        fr = int(l[0])
        if fr > 1:
            break
        UNKw.append(w)
    return UNKw

def PrepareData_UNK(ftrain,ngram,WordID,UNKw):
    ngram = ngram -1
    TrainingData=[]
    strLine=""
    for l in open(ftrain):
        l=l.strip()
        foundUNK = 0
        history = ["<s>","<s>","<s>","<s>"]
        for w in l.split():
            strLine = ""
            foundUNK=0
            for hw in history[len(history)-ngram:]:
                strLine = strLine+str(WordID[hw])+" "
                if hw in UNKw:
                    foundUNK=1
            if w  not in WordID:
                w="<UNK>"
                foundUNK=1;
            if w in UNKw:
                foundUNK=1
            strLine = strLine+str(WordID[w])
            TrainingData.append(strLine)

            strLine = ""
            i=0
            if foundUNK==1:
                for hw in history[len(history)-ngram:]:
                    if hw in UNKw:
                        strLine=strLine+str(WordID["<UNK>"])+' '
                    else:
                        strLine = strLine+str(WordID[hw])+" "
        
                if w in UNKw:
                    strLine=strLine+str(WordID["<UNK>"])
                else:
                    strLine=strLine+str(WordID[w])
                TrainingData.append(strLine)

            for ch in history[0:len(history)-1]:
                history[i]=history[i+1]
                i=i+1
            history[i]=w
        w = "</s>"
        strLine = ""
        for hw in history[len(history)-ngram:]:
            strLine = strLine+str(WordID[hw])+" "
        strLine = strLine+str(WordID[w])
        TrainingData.append(strLine)
    return TrainingData


def PrepareData(ftrain,ngram,WordID):
    ngram = ngram -1
    TrainingData=[]
    strLine=""
    N_unks = 0 
    for l in open(ftrain):
        l=l.strip()
        history = ["<s>","<s>","<s>","<s>"]
        for w in l.split():
            strLine = ""
            for hw in history[len(history)-ngram:]:
                strLine = strLine+str(WordID[hw])+" "
            if w  not in WordID:
                w="<UNK>"
	    if WordID[w] == WordID["<UNK>"]:
		N_unks = N_unks + 1 
            strLine = strLine+str(WordID[w])
            TrainingData.append(strLine)
            strLine = ""
            i=0
            for ch in history[0:len(history)-1]:
                history[i]=history[i+1]
                i=i+1
            history[i]=w
        w = "</s>"
        strLine = ""
        for hw in history[len(history)-ngram:]:
            strLine = strLine+str(WordID[hw])+" "
        strLine = strLine+str(WordID[w])
        TrainingData.append(strLine)
    print >> sys.stderr, "Number of UNKs encountered:", N_unks 
    return TrainingData

def CreateFeatData(ffeat,ftrain,fvocab,fvocabfreq,ngram,add_unk,use_unk):
    UNKw=[]
    if (fvocabfreq.strip()==""):
        print >> sys.stderr, "frequency file not given"
        add_unk=False
        use_unk=False
    else:
        UNKw = ReadFreqFile(fvocabfreq)
        if UNKw==[]:
            add_unk=False
            use_unk=False
    (WordID,printMapFile) = ReadVocabFile(fvocab)
    if printMapFile:
        fwrite = open(fvocab+".nnid",'w')
        for w in sorted(WordID, key=WordID.get):
            print >> fwrite, w,WordID[w]
    
    if add_unk:
        TrainingData = PrepareData_UNK(ftrain,ngram,WordID,UNKw)
    elif use_unk:
        TrainingData = PrepareData_UNK(ftrain,ngram,WordID,UNKw)
    else:
        TrainingData = PrepareData(ftrain,ngram,WordID)


def ReadWordID(infile):
    WordID = {}
    for l in open(infile):
	l=l.strip().split()
	w = l[0].strip()
	idx = int(l[1]	)
	WordID[w]=idx
    return WordID

def GetPerWordPenalty(WordID,ffreq):
    Penalty={}
    for l in open(ffreq):
	l=l.strip().split()
	w = l[1].strip()
        fr = int(l[0])
	if w in WordID:
	    Penalty[WordID[w]] = 2. /(1+math.log(fr,10)) 
	else:
	    print >> "No wordId for word",w
    Penalty[0] = 0.5
    Penalty[1] = 0.5
    Penalty[2] = 0.5 
    fpout = open("/tmp/jk",'w')
    print >> fpout, Penalty 
    return Penalty
	
	

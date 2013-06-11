import sys
sys.dont_write_bytecode = True
from numpy import zeros
import numpy
#import theano 
import scipy.io 
from Corpus import ReadWordID
from ConstructUnigram import ConstructUnigram 
def load_alldata_from_file(train,dev,test,ngram,N=4096):
    train_set = load_data_from_file(train,ngram,N,2000000)
    valid_set =load_data_from_file(dev,ngram,N,5000)
    test_set =load_data_from_file(test,ngram,N,5000)

    rval = [train_set, valid_set, test_set]

    return rval 


def load_alldata(trainList,devList,testList,ngram,N=4096):
    train_set = load_data(trainList,ngram,N,2000000)
    valid_set =load_data(devList,ngram,N,5000)
    test_set =load_data(testList,ngram,N,5000)

    rval = [train_set, valid_set, test_set]
    
    return rval


def load_data_from_file(dataset,ngram,N=4096,maxS=200000):
    DataA={}
    for i in range(ngram-1):
        DataA[i]=[]

    TargetA=[]
    tot=0
    for l in open(dataset):
        tot+=1
        if tot%10000==0:
            print >> sys.stderr,tot,"...",
        l=l.strip().split()
        target = int(l[len(l)-1])
        if target>=N:
            target = 2

        l=l[0:len(l)-1]
        il = map(int,l)
        idx=0
        for i in il:
            if i >=N:
                i=2
            DataA[idx].append(i)
            idx=idx+1
        TargetA.append(target)
        if tot == maxS:
            break

    TargetA1 = numpy.asarray(TargetA,dtype=numpy.int32).reshape(-1)
    print >> sys.stderr, "\nRead",tot,ngram,"-grams"
    return [DataA[i] for i in range(ngram-1)],TargetA1


def load_data(dataList,ngram,N=4096,maxS=200000):
    DataA={}
    for i in range(ngram-1):
        DataA[i]=[]

    TargetA=[]
    tot=0
    for l in dataList:
        tot+=1
        if tot%10000==0:
            print >> sys.stderr,tot,"...",
        l=l.strip().split()
        target = int(l[len(l)-1])
        if target>=N: #check that word id < input layer size
            target = 2

        l=l[0:len(l)-1]
        il = map(int,l)
        idx=0
        for i in il:
            if i >=N:
                i=2
            DataA[idx].append(i)
            idx=idx+1
        TargetA.append(target)
        if tot == maxS:
            break

    TargetA1 = numpy.asarray(TargetA,dtype=numpy.int32).reshape(-1)
    print >> sys.stderr, "\nRead",tot,ngram,"-grams"
    return [DataA[i] for i in range(ngram-1)],TargetA1


def load_params_matlab(fparam):
    print >> sys.stderr, "Reading matlab files from dir : ",fparam
    pW  = scipy.io.loadmat(fparam+'/pW.mat')['pW']
    hW  = scipy.io.loadmat(fparam+'/hW.mat')['hW']
    hB  =  numpy.asarray(scipy.io.loadmat(fparam+'/hB.mat')['hB']).reshape(-1)
    lB  = numpy.asarray(scipy.io.loadmat(fparam+'/lB.mat')['lB']).reshape(-1)
    #lB2 = numpy.asarray(scipy.io.loadmat(fparam+'/lB2.mat')['lB2']).reshape(-1)
    lW  = scipy.io.loadmat(fparam+'/lW.mat')['lW']
    #lW2 = scipy.io.loadmat(fparam+'/lW2.mat')['lW2']
    return (pW,hW,hB,lB,lW)

def load_params(fparam):
    W = {}
    w=[]
    i=-1
    for l in open(fparam):
        l=l.strip()
        if l=="<W>":
            W[i+1]=[]
            w=[]
            i=i+1
            continue
        if l=="</W>":
            if len(w)>0:

                W[i].append(w)
            w=[]
            continue
        else:

            if l.find('[')>=0:
                if len(w)>0:
                    W[i].append(w)
                w=[]
                l=l.replace('[','')
            if l.find(']')>=0:

                l=l.replace('[','')
            if l.find(']')>=0:
                l=l.replace(']','')
            if i>=2:
                if len(w)>0:
                    W[i].append(w)
                w=[]
            l=l.split()
            for weight in l:
                w.append(float(weight))

    name=['pW','hW','hB','lB','lB2','lW','lW2']
    print >> sys.stderr, "loading params..."
    for i in range(7):
        w_values = numpy.asarray((W[i]),dtype=theano.config.floatX)
        if len(w_values[0])==1:
            w_values = numpy.asarray(w_values).reshape(-1)

        W[i]=w_values
        print >> sys.stderr, name[i],w_values.shape
    return ((W[0],W[1],W[2],W[3],W[4],W[5],W[6]))# (pW,hW,hB,lB,lB2,lW,lW2)


def write_params_matlab(fparam,pW,hW,hB,lB,lW,lB2=None,lW2=None):
    scipy.io.savemat(fparam+'/pW.mat',mdict={'pW':pW},format='4')
    scipy.io.savemat(fparam+'/hW.mat',mdict={'hW':hW},format='4')
    scipy.io.savemat(fparam+'/hB.mat',mdict={'hB':hB},format='4')
    scipy.io.savemat(fparam+'/lB.mat',mdict={'lB':lB},format='4')
    #scipy.io.savemat(fparam+'/lB2.mat',mdict={'lB2':lB2},format='4')
    scipy.io.savemat(fparam+'/lW.mat',mdict={'lW':lW},format='4')
    #scipy.io.savemat(fparam+'/lW2.mat',mdict={'lW2':lW2},format='4')
    scipy.io.savemat(fparam+'/All.mat',mdict={'pW':pW,'hW':hW,'hB':hB,'lW':lW,'lB':lB},format='5')
#deprected function 
def write_params(param_file,pW,hW,hB,lB,lW,lB2=None,lW2=None):
    fparam = open(param_file,"w")

    for w in (pW,hW,hB,lB,lB2):
        print >> fparam,"<W>"
        for wi in numpy.asarray(w):
            print >> fparam, wi
        print >> fparam,"</W>"

    print >> fparam,"<W>"
    for wi in numpy.asarray(lW):
        for wij in wi:
            print >> fparam, wij,
        print >> fparam, ''
    print >> fparam, "</W>"
    print >> fparam,"<W>"
    for wi in numpy.asarray(lW2):
        for wij in wi:
            print >> fparam, wij,
        print >> fparam, ''
    print >> fparam, "</W>"

def WriteData(DataList, filename):
    print >> sys.stderr, "Writing to file:",filename,", No of lines:", len(DataList)
    fwrite = open(filename,'w')
    for l in DataList:
        print >> fwrite, l 
    fwrite.close()

def read_machine(paramdir):
    infile = paramdir+"/mach.desc"
    for l in open(infile):
        l=l.strip().lower()
	print l
        if l.find("projection")>=0:
            l=l.strip().split(':')
            P = int(l[1].strip())
	    continue
        if l.find("vocab size")>=0:
            l=l.strip().split(':')
            N = int(l[1].strip())
            continue
        if l.find("hidden")>=0:
            l=l.strip().split(':')
            H = int(l[1].strip())
            continue
	if l.find("ngram")>=0:
            l=l.strip().split(':')
            ngram = int(l[1].strip())
            continue
	if l.find("map")>=0:
            l=l.strip().split(':')
            mapfile = l[1].strip()
	    continue
	if l.find("feature")>=0:
	    l=l.strip().split(":")	
	    n_feats = int(l[1].strip())
 
    WordID = ReadWordID(mapfile)
    return ngram,n_feats,N,P,H,WordID

def write_janus_LM(fvocab,fparam,fsrilm):
    fout = fparam+"/unigram"
    ConstructUnigram(fvocab,fsrilm,fout)

if __name__ == '__main__':
   pW,hW,hB,lB,lW = load_params_matlab(sys.argv[1])	
   write_params_matlab(sys.argv[2],pW,hW,hB,lB,lW) 

import sys
sys.dont_write_bytecode = True
from numpy import zeros
import numpy
import os
#import theano 
import scipy.io 
from Corpus import ReadWordID
from ConstructUnigram import ConstructUnigram 
def load_alldata_from_file(train,dev,test,ngram,N=4096,unkid=2):
    train_set = load_data_from_file(train,ngram,N,2000000,unkid)
    valid_set =load_data_from_file(dev,ngram,N,5000,unkid)
    test_set =load_data_from_file(test,ngram,N,5000,unkid)

    rval = [train_set, valid_set, test_set]

    return rval 


def load_alldata(trainList,devList,testList,ngram,N=4096,unkid=2):
    train_set = load_data(trainList,ngram,N,2000000,unkid)
    valid_set =load_data(devList,ngram,N,5000,unkid)
    test_set =load_data(testList,ngram,N,5000,unkid)

    rval = [train_set, valid_set, test_set]
    
    return rval


def load_data_from_file(dataset,ngram,N=4096,maxS=200000,unkid=2):
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
	    if unkid >= 0 :
                target = unkid 

        l=l[0:len(l)-1]
        il = map(int,l)
        idx=0
        for i in il:
            if i >=N:
		if unkid >=0:
                   i=unkid 
            DataA[idx].append(i)
            idx=idx+1
        TargetA.append(target)
        if tot == maxS:
            break

    TargetA1 = numpy.asarray(TargetA,dtype=numpy.int32).reshape(-1)
    print >> sys.stderr, "\nRead",tot,ngram,"-grams"
    return [DataA[i] for i in range(ngram-1)],TargetA1


def load_data(dataList,ngram,N=4096,maxS=200000,unkid=2):
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
            target = unkid

        l=l[0:len(l)-1]
        il = map(int,l)
        idx=0
        for i in il:
            if i >=N:
                i=unkid
            DataA[idx].append(i)
            idx=idx+1
        TargetA.append(target)
        if tot == maxS:
            break

    TargetA1 = numpy.asarray(TargetA,dtype=numpy.int32).reshape(-1)
    print >> sys.stderr, "\nRead",tot,ngram,"-grams"
    return [DataA[i] for i in range(ngram-1)],TargetA1


def load_params_matlab(fparam,number_hidden_layer =1):
    print >> sys.stderr, "Reading matlab files from dir : ",fparam
    pW  = scipy.io.loadmat(fparam+'/pW.mat')['pW']
    hW  = scipy.io.loadmat(fparam+'/hW.mat')['hW']
    hB  =  numpy.asarray(scipy.io.loadmat(fparam+'/hB.mat')['hB']).reshape(-1)
    lB  = numpy.asarray(scipy.io.loadmat(fparam+'/lB.mat')['lB']).reshape(-1)
    #lB2 = numpy.asarray(scipy.io.loadmat(fparam+'/lB2.mat')['lB2']).reshape(-1)
    lW  = scipy.io.loadmat(fparam+'/lW.mat')['lW']
    #lW2 = scipy.io.loadmat(fparam+'/lW2.mat')['lW2']
    return (pW,hW,hB,lB,lW)

def load_params_matlab_multi(fparam,number_hidden_layer =1):
    print >> sys.stderr, "Reading matlab files from dir : ",fparam
    pW  = scipy.io.loadmat(fparam+'/All.mat')['pW']
    scipy.io.savemat(fparam+'/pW.mat',mdict={'pW':pW},format='4')
    hW=[]
    hB=[]
    for i in range(number_hidden_layer):
        if os.path.isfile((fparam+'/hW'+str(i)+'.mat')):
	   hweight = scipy.io.loadmat(fparam+'/hW'+str(i)+'.mat')['hW'+str(i)]
	   hbias = numpy.asarray(scipy.io.loadmat(fparam+'/hB'+str(i)+'.mat')['hB'+str(i)]).reshape(-1)   
	else:
	   hweight  = scipy.io.loadmat(fparam+'/hW.mat').items()[i][1]#['hW']
	   hbias  =  numpy.asarray(scipy.io.loadmat(fparam+'/hB.mat').items()[i][1]).reshape(-1) #['hB']).reshape(-1)
	hW.append(hweight)
	hB.append(hbias)

    lB  = numpy.asarray(scipy.io.loadmat(fparam+'/lB.mat')['lB']).reshape(-1)
    #lB2 = numpy.asarray(scipy.io.loadmat(fparam+'/lB2.mat')['lB2']).reshape(-1)
    lW  = scipy.io.loadmat(fparam+'/lW.mat')['lW']
    #lW2 = scipy.io.loadmat(fparam+'/lW2.mat')['lW2']
    print >> sys.stderr, 'pW',pW.shape , 'lW', lW.shape , 'lB',lB.shape 
    print >> sys.stderr, 'hW', hW[0].shape , 'hB', hB[0].shape
    
    return (pW,hW,hB,lW,lB)

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
    WriteDict={}
    WriteDict['pW'] = pW
    scipy.io.savemat(fparam+'/pW.mat',mdict={'pW':pW},format='4')
    i=0
    for hweights in hW:
	WriteDict['hW'+str(i)]=hweights
    	scipy.io.savemat(fparam+'/hW'+str(i)+'.mat',mdict={'hw'+str(i):hweights},format='4')
	i=i+1
    i=0
    for hbias in hB:
	WriteDict['hB'+str(i)]=hbias
    	scipy.io.savemat(fparam+'/hB'+str(i)+'.mat',mdict={'hB'+str(i):hbias},format='4')
	i=i+1 
    WriteDict['lB'] = lB
    scipy.io.savemat(fparam+'/lB.mat',mdict={'lB':lB},format='4')
    #scipy.io.savemat(fparam+'/lB2.mat',mdict={'lB2':lB2},format='4')
    WriteDict['lW'] = lW
    scipy.io.savemat(fparam+'/lW.mat',mdict={'lW':lW},format='4')
    #scipy.io.savemat(fparam+'/lW2.mat',mdict={'lW2':lW2},format='4')
    
    scipy.io.savemat(fparam+'/All.mat',mdict=WriteDict,format='5') #{'pW':pW,'hW':hW,'hB':hB,'lW':lW,'lB':lB},format='5')
#deprected function 
def write_all_params_matlab(fparam,params,number_hidden_layer=1):
    p=0
    pW = params[p] 
    p=p+1 
    hW = []; hB = []
    if len(params[p])==number_hidden_layer:
	isList= 1
    else:
	isList = 0
    for i in range(number_hidden_layer):
        if isList==0:
	    hW.append(params[p])
            p=p+1
	    hB.append(params[p])
	    p=p+1
        else:
	    hW.append(params[p][i])
	    hB.append(params[p+1][i])
    if isList==1:
	p = p + 2  
    lW = params[p]; p=p+1
    lB = params[p]; p=p+1

    WriteDict={}
    WriteDict['pW'] = pW
    scipy.io.savemat(fparam+'/pW.mat',mdict={'pW':pW},format='4')
    i=0
    for hweights in hW:
 	if number_hidden_layer > 1:
            WriteDict['hW'+str(i)]=hweights
	else:
            WriteDict['hW'] = hweights
        scipy.io.savemat(fparam+'/hW'+str(i)+'.mat',mdict={'hW'+str(i):hweights},format='4')
        i=i+1
    i=0
    for hbias in hB:
	if number_hidden_layer > 1:
            WriteDict['hB'+str(i)]=hbias
	else:
	    WriteDict['hB']=hbias
        scipy.io.savemat(fparam+'/hB'+str(i)+'.mat',mdict={'hB'+str(i):hbias},format='4')
        i=i+1
    WriteDict['lB'] = lB
    scipy.io.savemat(fparam+'/lB.mat',mdict={'lB':lB},format='4')
    #scipy.io.savemat(fparam+'/lB2.mat',mdict={'lB2':lB2},format='4')
    WriteDict['lW'] = lW
    scipy.io.savemat(fparam+'/lW.mat',mdict={'lW':lW},format='4')
    #scipy.io.savemat(fparam+'/lW2.mat',mdict={'lW2':lW2},format='4')

    scipy.io.savemat(fparam+'/All.mat',mdict=WriteDict,format='5')

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
    number_hidden_layer = 1
    for l in open(infile):
	x = l
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
        if l.find("hidden layer:")>=0:
            l=l.strip().split(':')
            H = int(l[1].strip())
            continue
	if l.find("ngram")>=0:
            l=l.strip().split(':')
            ngram = int(l[1].strip())
            continue
	if l.find("map")>=0:
            x=x.strip().split(':')
            mapfile = x[1].strip()
	    continue
	if l.find("feature")>=0:
	    l=l.strip().split(":")	
	    n_feats = int(l[1].strip())
	    continue 
 	if l.find("number of hidden layers")>=0:
	    l=l.strip().split(":")
            number_hidden_layer = int(l[1].strip())
	
    WordID = ReadWordID(mapfile)
    return ngram,n_feats,N,P,H,number_hidden_layer,WordID

def write_janus_LM(fvocab,fparam,fsrilm):
    fout = fparam+"/unigram"
    ConstructUnigram(fvocab,fsrilm,fout)

if __name__ == '__main__':
   pW,hW,hB,lB,lW = load_params_matlab(sys.argv[1])	
   write_params_matlab(sys.argv[2],pW,hW,hB,lB,lW) 

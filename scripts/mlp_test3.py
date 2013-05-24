import sys 
sys.dont_write_bytecode = True
import cPickle
import gzip
import os
import sys
import time

import numpy
import gc
import theano
import theano.tensor as T
from numpy import * 
from NNLMio import load_params_matlab
from logistic_sgd7 import LogisticRegression

class ProjectionLayer(object):
    def __init__(self, rng, input, history1, n_in, n_out, N=4096, W=None,
                 sparse=None,activation=None):
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (N*n_in + n_out)),
                    high=numpy.sqrt(6. / (N*n_in + n_out)),
                    size=(N*n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            w_values=W
            W = theano.shared(value=w_values, name='W',borrow=True)
       
        #self.sparse= sparse
        self.W = W
        
        lin_output = T.concatenate((T.dot(self.input, self.W),T.dot(history1,self.W)),axis=1)
        self.output = lin_output #(lin_output if activation is None
                    #   else activation(lin_output))
        # parameters of the model
        self.params = [self.W]



class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            W = theano.shared(value=W,name='W',borrow=True)
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            b = theano.shared(value=b, name='b', borrow=True)
        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):

    def __init__(self, rng, input, history1, n_in,n_P, n_hidden, n_out, ngram, Voc=4096,pW=None,hW=None,hB=None,lW=None,lB=None,lW2=None,lB2=None):
        
        self.projectionLayer = ProjectionLayer(rng=rng, input=input, history1=history1, n_in=n_in, n_out=n_P, W=pW,N=Voc,activation=None)
        self.hiddenLayer = HiddenLayer(rng=rng, input=self.projectionLayer.output,
                                       n_in=n_P*ngram, n_out=n_hidden,
                                       activation=T.tanh, W=hW,b=hB)
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,W=lW,b=lB,W2=lW2,b2=lB2)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum() \
                + abs(self.logRegressionLayer.W2).sum()


        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()


        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        self.tot_ppl = self.logRegressionLayer.tot_ppl
        self.get_p_y_given_x = self.logRegressionLayer.get_p_y_given_x2
        # the parameters of the model are the parameters of the two layer it is
        # made out o
        
        self.params = self.projectionLayer.params + self.hiddenLayer.params + self.logRegressionLayer.params  
        
    def get_params_pW(self):
        return (self.projectionLayer.W)
    def get_params_hW(self):
        return self.hiddenLayer.W
    def get_params_hb(self):
        return self.hiddenLayer.b
    def get_params_lW(self):
        return self.logRegressionLayer.W 
    def get_params_lb(self):
        return self.logRegressionLayer.b
    def get_params_lW2(self):
        return self.logRegressionLayer.W2
    def get_params_lb2(self):
        return self.logRegressionLayer.b2

def convert_to_sparse(x,N=4096):
    data=[]
    for i in x:
        y = zeros((N),dtype=theano.config.floatX)
        if i >=N:
            i=2
        y[i]=1
        z=y
        data.append(z)
    return data 

def load_data2(dataset,n_in,N=4096,low=0,maxS=10000):

    DataA={}#numpy.array(0)
    DataA[0]=[]
    DataA[1]=[]
    TargetA=[]
    tot=0
    for l in open(dataset):
        tot+=1
	if tot<low:
	    continue
        if tot%1000==0:
            print tot
        l=l.strip().split()
        target = int(l[len(l)-1])
        if target>=N:
            target = 2
        
        l=l[0:len(l)-1]
        il = map(int,l)
        y = zeros((N),dtype=theano.config.floatX)
        z=[]
	idx=0
        for i in il:
            y = zeros((N),dtype=theano.config.floatX)
            if i >=N:
                i=2
            y[i]=1
            z=i
        #z = numpy.asarray(z).reshape(-1)
            DataA[idx].append(z)
	    idx=idx+1
        TargetA.append(target)
        if tot == maxS:
            break
    
    TargetA1 = numpy.asarray(TargetA,dtype=numpy.int32).reshape(-1)
    return ((DataA[0],DataA[1],TargetA1))

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
                l=l.replace(']','')
            if i>=2:
                if len(w)>0:
                    W[i].append(w)
                w=[]
            l=l.split()
            for weight in l:                                                                                                                 
                w.append(float(weight))
    
    name=['pW','hW','hB','lB','lB2','lW','lW2']
    print "loading params..."
    for i in range(7):
        w_values = numpy.asarray((W[i]),dtype=theano.config.floatX)
        if len(w_values[0])==1:
            w_values = numpy.asarray(w_values).reshape(-1)
        #w = theano.shared(value=w_values, name=name[i], borrow=True)
        W[i]=w_values
        print name[i],w_values.shape#),len(w_values[0])
        #dummy = numpy.zeros((10000,50)
 
def shared_dataset(data_xy, borrow=False):
    """ Function that loads the dataset into shared variables
    
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
    data_x, data_x1, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_x1 = theano.shared(numpy.asarray(data_x1,
                                           dtype=theano.config.floatX),
                             borrow=borrow)

    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, shared_x1, T.cast(shared_y, 'int32')


def test_mlp(learning_rate=0.005, L1_reg=0.00, L2_reg=0.0001, n_epochs=50, batch_size=50):
    N = 10000
    ngram = 2
    copy_size=10000
    N,P,paramindir, test, outfile,a,b= sys.argv[1:] # data/train.txt.fmt, out/y.probs
    N = int(N)
    P = int(P)
    outfile = outfile+'.'+a
    dataset = load_data2(test,ngram,N,float(a),float(b))
    pW,hW,hB,lB,lB2,lW,lW2 = load_params_matlab(paramindir) 
    ntest_set_full_x,ntest_set_full_x1,ntest_set_full_y = dataset
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    x1 = T.matrix('x1')
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    rng = numpy.random.RandomState(1234)
    classifier = MLP(rng=rng, input=x, history1=x1, n_in=1,n_P=P, n_hidden=500, n_out=N, Voc=N,pW=pW,hW=hW,hB=hB,lW=lW,lW2=lW2,lB=lB,lB2=lB2,ngram=ngram)
    probs_model_full = theano.function(inputs=[x,x1, y], outputs=[classifier.get_p_y_given_x(y)])

    foutPF=open(outfile,'w')
    gc.collect()
    t = len(ntest_set_full_y)
    if copy_size > t:
        test_full_batches=1
        copy_size=t
    else:
        test_full_batches = t/copy_size + 1
    for i in range(test_full_batches):
        xfull = numpy.asarray(convert_to_sparse(ntest_set_full_x[i*copy_size: (i+1)*copy_size],N))
        x1full = numpy.asarray(convert_to_sparse(ntest_set_full_x1[i*copy_size: (i+1)*copy_size],N))
        yfull = ntest_set_full_y[i*copy_size : (i+1)*copy_size]
        scores = probs_model_full(xfull,x1full,yfull)
        for yl in scores:
            for xi in yl:
                print >> foutPF, xi
        
        gc.collect()
        
    
if __name__ == '__main__':
    test_mlp()

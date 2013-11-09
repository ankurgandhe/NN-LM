#With Proj units as input 
"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
B
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

i"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy
import scipy.io
import gc
import theano
import theano.tensor as T
from numpy import * 
from theano import sparse as Tsparse 
import scipy.sparse as sp 
from logistic_sgd import LogisticRegression
from NNLMio import read_machine,load_data,load_params_matlab_multi,load_data_from_file, write_all_params_matlab
from Corpus import CreateData

copy_size=75000
ProjectFeat=1
UseDirect=0
class ProjectionLayer_ngram(object):
    def __init__(self, rng, input, nhistory, feature,n_feat, n_in, n_out, N=4096, W=None,
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

        self.W = W
	if ProjectFeat and n_feat>0:
	    if not UseDirect:
	    	self.input_feat = T.concatenate((self.input,feature[:,0:n_feat]),axis=1)
            	lin_output = T.dot(self.input_feat, self.W)   
	    else:
                self.input_feat = T.concatenate((self.input,feature[:,0:n_feat]),axis=1)
		lin_output = self.W[T.arrange(self.input_feat.shape[0])]
	else:
	    if not UseDirect:
            	lin_output = T.dot(self.input, self.W)
	    else:
		lin_output = self.W[T.arange(self.input)]

        for history in nhistory:
	   if not UseDirect: 
	   	if ProjectFeat and n_feat>0:
		    history = T.concatenate((history,feature[:,n_feat:n_feat*2]),axis=1)
            	lin_output = T.concatenate((lin_output,T.dot(history,self.W)),axis=1)
	   else:
                #implemetn features
                lin_output = T.concatenate((lin_output,self.W[T.arange(history)]),axis=1)
         
        if n_feat==0:
            self.output = lin_output
        else:
	    if ProjectFeat==1:
                self.output = lin_output
	    else:
            	self.output = T.concatenate((lin_output,feature),axis=1)
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
                W_valueds *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
        #else:
        #    W = theano.shared(value=W,name='W',borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        #else:
        #    b = theano.shared(value=b, name='b', borrow=True)
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

    def copy_weights(self,weight,bias):
	self.W = weight
	self.b = bias 

def initialize_weights(rng,pW,pb,n_in,n_out,activation):
  if pW is None:
      W_values = numpy.asarray(rng.uniform(
              low=-numpy.sqrt(6. / (n_in + n_out)),
              high=numpy.sqrt(6. / (n_in + n_out)),
              size=(n_in, n_out)), dtype=theano.config.floatX)
      if activation == theano.tensor.nnet.sigmoid:
          W_valueds *= 4
          
      W = theano.shared(value=W_values, name='W', borrow=True)
  else:
      W = theano.shared(value=pW,name='W', borrow=True)

  if pb is None:
      b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)
  else:
      b = theano.shared(value=pb , name='W', borrow=True)

  return W,b 

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activation
    """
    
    def __init__(self, rng, input, nhistory, feature,n_features, n_in,n_P, n_hidden,number_hidden_layer, n_out, ngram=3, Voc=4096,
		 pW=None,hW=None,hB=None,lW=None,lB=None,shared=False,train_projection=True,new_weights=True):
        if ProjectFeat:
	    Voc = Voc + n_features
	# if not train, ten we dirctly use pW as feats
	if not train_projection:	
	    self.pW = theano.shared(value=pW, name='b', borrow=True)
 
	self.train_projection = train_projection 
	if train_projection:
            self.projectionLayer = ProjectionLayer_ngram(rng=rng, input=input, nhistory=nhistory, 
	
			   feature=feature, n_feat = n_features, n_in=1, n_out=n_P, W=pW, N=Voc,activation=None)
	self.HiddenLayers=[]
	self.nH = number_hidden_layer
	self.hsharedW=[]
	self.hsharedb=[]
	if ProjectFeat:
	    hidden_insize = n_P*(ngram-1)
	else:
	    hidden_insize = n_P*(ngram-1)+ n_features*(ngram-1)

	for i in range(number_hidden_layer): 
	    if hW==None or new_weights==True:
		if hW!=None:
		    print >> sys.stderr, "Creating new hidden weights" 
		if i==0:
		    hweights,hbias = initialize_weights(rng,None,None,hidden_insize,n_hidden,activation=T.tanh)
		else:
	 	    hweights,hbias = initialize_weights(rng,None,None,n_hidden,n_hidden,activation=T.tanh)
		self.hsharedW.append(hweights)
		self.hsharedb.append(hbias) 
		#hweights = None
		#hbias = None 
	    else:
                if shared==False:
		     hweights,hbias = initialize_weights(rng,hW[i],hB[i],hidden_insize,n_hidden,activation=T.tanh)
		     self.hsharedW.append(hweights)
                     self.hsharedb.append(hbias)
	 	else:
		     hweights = hW[i]
		     hbias = hB[i]
	    if i==0:
		if train_projection:
             	    hiddenLayer = HiddenLayer(rng=rng, input=self.projectionLayer.output,
             	                          n_in=hidden_insize, n_out=n_hidden,
                	                       activation=T.tanh,W=hweights,b=hbias)
		else:
                    hiddenInput = input
                    for xhistory in nhistory:
                        hiddenInput = T.concatenate((hiddenInput,xhistory),axis=1)
		    hiddenLayer = HiddenLayer(rng=rng, input=hiddenInput,
                                          n_in=hidden_insize, n_out=n_hidden,
                                               activation=T.tanh,W=hweights,b=hbias)

	    else:
		hiddenLayer = HiddenLayer(rng=rng, input=self.HiddenLayers[i-1].output,
                                          n_in=n_hidden, n_out=n_hidden,
                                               activation=T.tanh,W=hweights,b=hbias)	
	    self.HiddenLayers.append(hiddenLayer)
        # The logistic regression layer gets as input the hidden units++++++++++++++++++++
        # of the hidden layer

	if new_weights and lW!=None:
	    print >> sys.stderr,"Creating new logistic weights"
	    self.logRegressionLayer = LogisticRegression(input=self.HiddenLayers[number_hidden_layer-1].output,
                                                     n_in=n_hidden,
                                                     n_out=n_out)
	else:
            self.logRegressionLayer = LogisticRegression(input=self.HiddenLayers[number_hidden_layer-1].output,
                                                     n_in=n_hidden,
                                                     n_out=n_out,W=lW,b=lB)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
	self.L1 = abs(self.logRegressionLayer.W).sum()
	for i in range(number_hidden_layer):
	    self.L1  = self.L1 + abs(self.HiddenLayers[i].W).sum()
        #self.L1 = abs(self.hiddenLayer.W).sum() \
        #        + abs(self.logRegressionLayer.W).sum() 


        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
	self.L2_sqr = (self.logRegressionLayer.W ** 2).sum() 
        for i in range(number_hidden_layer):
	    self.L2_sqr = self.L2_sqr + (self.HiddenLayers[i].W ** 2).sum()
        #self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
        #            + (self.logRegressionLayer.W ** 2).sum() \


        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors

        self.tot_ppl = self.logRegressionLayer.tot_ppl
        self.get_p_y_given_x = self.logRegressionLayer.get_p_y_given_x2
        self.get_p_y_given_x_all = self.logRegressionLayer.get_p_y_given_x #logRegressionLayer.get_p_y_given_x
        # the parameters of the model are the parameters of the two layer it is
        # made out o
        
	if not UseDirect and train_projection:
            self.params = self.projectionLayer.params #+ #self.hiddenLayer.params + self.logRegressionLayer.params  
	    for i in range(number_hidden_layer):
	    	self.params = self.params + self.HiddenLayers[i].params 
	    self.params = self.params + self.logRegressionLayer.params
	else:
            #self.params = self.projectionLayer.params #+ #self.hiddenLayer.params + self.logRegressionLayer.params
            self.params = self.logRegressionLayer.params
            for i in range(number_hidden_layer):
                self.params = self.params + self.HiddenLayers[i].params
            #self.params = self.params + self.logRegressionLayer.params


    def copy_hWeights(self, hWeights,bias):
	self.HiddenLayers[0].copy_weights(hWeights,bias)
        
    def get_params_pW(self):
	if self.train_projection:
            return (self.projectionLayer.W)
	else:
	    return self.pW
    def get_params_hW(self,i):
        return self.HiddenLayers[i].W #for i in range(self.nH)]#self.hiddenLayer.W
    def get_params_hb(self,i):
        return self.HiddenLayers[i].b #for i in range(self.nH)]#self.hiddenLayer.b
    def get_params_lW(self):
        return self.logRegressionLayer.W 
    def get_params_lb(self):
        return self.logRegressionLayer.b

def convert_to_sparse(x,N=4096):
    data = zeros((len(x),N),dtype=theano.config.floatX)
    n = 0
    for i in x:
        if i >=N:
            i=2
        data[n][i] = 1
	n = n+1
    return data

def convert_to_sparse_matrix(x,N=4096):
    data = []
    row = []
    col = [] 
    n=0
    for i in x:
        if i >=N:
            i=2
        data.append(1)
        row.append(n)
        col.append(i)
        n=n+1
    data_matrix = sp.csr_matrix( (data,(row,col)), shape=(len(x),N),dtype=theano.config.floatX )
    return data_matrix 

def convert_to_projection(x,pW,N=4096):
    n=0
    projectionList=[]
    for i in x:
        if i >=N:
            i=2
	projectionList.append(pW[i])
        n=n+1
    data_matrix = numpy.asarray(projectionList,dtype=theano.config.floatX)
    #sp.csr_matrix( (data,(row,col)), shape=(len(x),N),dtype=theano.config.floatX )
    return data_matrix


def convert_to_sparse_combine(Listx,N=4096,a=0,b=1e20):
    data=[]
    b = min(b,len(Listx[0]))
    data = zeros((b-a,N),dtype=theano.config.floatX)

    for n in range(a,b):
        for x in Listx:
            i = x[n]
            if i >=N:
                i=2
            data[n][i]=1
        
    return data


def convert_to_sparse_matrix_combine(Listx,N=4096,a=0,b=1e20):
    data = []
    row = []
    col = []
    b = min(b,len(Listx[0]))
   
    for n in range(a,b):
        for x in Listx:
            i = x[n]
            if i >=N:
                i=2
	    data.append(1)
            row.append(n)
            col.append(i)
    data_matrix = sp.csr_matrix( (data,(row,col)), shape=(len(x),N),dtype=theano.config.floatX )
    return data_matrix

def convert_to_sparse_matrix_feature(Listx,N=4096,a=0,b=1e20):
    data = []
    row = []
    col = []
    b = min(b,len(Listx[0]))
    # for each training example 
    for n in range(a,b):
	c=0
	# for each xk-1 xk-2 ... 
        for x in Listx:
	    # get current n's context
            i = x[n]
	    while( i >=N):
                # to convert bigger into many smaller integers
                f = i % 256 # 8 bit
                data.append(1)
		row.append(n)
		col.append(c*N+f)
                i = i >> 8 ; # shift by 8 bit

            #if i >=N:
	    #	print >> sys.stderr, "How is this possible!!!" 
            #    i=0
            data.append(1)
            row.append(n)
            col.append(c*N + i )
	    c = c+1
    #print "ROW",row
    #print "COL", col 
    data_matrix = sp.csr_matrix( (data,(row,col)), shape=(len(x),N*c),dtype=theano.config.floatX )
    return data_matrix

def shared_data(data,type="float",sparse=False,borrow=False):
    data_x = data
    if sparse==False:
        shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),
                                 borrow=borrow)
    else:
        shared_x = theano.shared(data_x,
                                 borrow=borrow)

    if type=="float":
        return shared_x 
    elif type=="int"or type=="int32":
        return T.cast(shared_x, 'int32')

	
def GetPenaltyVector(y,penalty,Wids,share=True):
    # check if its bigram penalty
    # bigram penalty is stored in lists
    if isinstance(Wids,list):
	return GetBigramPenaltyVector(0,len(y),Wids,share)
    # Wids = list of words that need to be penalized with penalty
    vec_penalty  = zeros(len(y ),dtype=theano.config.floatX)
    #vec_penalty2  = zeros(len(y ),dtype=theano.config.floatX) # just for stats
    #vec_penalty = [] 
    i  = 0
    count = 0
    max_wid = 10000
    '''if Penalty=={}:
	GetPerWordPenalty();
    Penalty={}
    for x in range(max_wid*2):
	if x in Wids:
	    Penalty[x] = penalty
	else:
	    Penalty[x] = 1
    '''
    Penalty = Wids 
    while i < len(y): #vec_penalty):
        #if 1 : #y[i] in Wids: 
        #    vec_penalty.append(penalty) #[i]=penalty
	#    count = count + 1
        #else:
        #vec_penalty.append(penalty)#[i]=1
	vec_penalty[i] = Penalty[y[i]] #.append(Penalty[y[i]])
        #vec_penalty2[i] = Penalty[y[i]] -1 #.append(Penalty[y[i]])
	if vec_penalty[i]!=1:
	    count = count + 1
	if i%5000==0:
	    print >> sys.stderr, i,",",
	i=i+1 
    if share: 
    	vec_penalty_shared  =  theano.shared(vec_penalty, borrow=True)
    else:
	vec_penalty_shared = vec_penalty #numpy.asarray(vec_penalty,dtype=theano.config.floatX) 
    #print >> sys.stderr, "Penalized ", numpy.sum(vec_penalty)/12 #sum(Penalty.values())," words"
    print >> sys.stderr, "\nPenalized" , count , "examples" 
    return vec_penalty_shared

def GetBigramPenaltyVector(ystart,yend,penalty,share=True):
    lcount =0	
    pen = 4
    vec_penalty  = zeros(yend-ystart,dtype=theano.config.floatX)
    j=0
    for i in range(ystart,yend):
	vec_penalty[j] = 1 + penalty[i]*pen
	j = j + 1

    if share:
        vec_penalty_shared  =  theano.shared(vec_penalty, borrow=True)
    else:
        vec_penalty_shared = vec_penalty #numpy.asarray(vec_penalty,dtype=theano.config.floatX)
    
    print >> sys.stderr, "Bigram Penlaty enabled. total of ", sum(penalty)  
    return vec_penalty_shared 
	
	


def shuffle_in_unison_inplace(a, b):
    assert len(a[0]) == len(b)
    p = numpy.random.shuffle( numpy.arange(len(b)))
    i =0 	
    newa = [] 
    for x in a:
	newa.append(x[p])
	i = i+1 
    return newa, b


class TrainMLP(object):

    def __init__(self,NNLMdata,NNLMFeatData,OldParams,ngram,n_feats,n_unk,N,P,H,number_hidden_layer,learning_rate, L1_reg, L2_reg, 
                 n_epochs,batch_size,adaptive_learning_rate,fparam,gpu_copy_size,spl_words,train_projection,new_weights,
		 hWshared=None,hBshared=None,shared=False):

        self.learning_rate0 = learning_rate
	self.ngram = ngram 
	self.n_unk = n_unk
	self.n_feats = n_feats 
	self.L1_reg = L1_reg 
	self.L2_reg = L2_reg 
	self.L2_struct = 0.1
	self.train_projection = train_projection 
	self.counter=0
        # make sure pW is available when we are not trnaining projection
	if not train_projection and not OldParams:
            print >> sys.stderr, "For pW to be not trianed, we need an initial model. Exiting"
            sys.exit(0)

        if OldParams:
            pW,hW,hB,lW,lB = OldParams
            self.learning_rate = 0.05
        else:
            pW = hW = hB = lB =  lW =  None

	self.pW = pW 

        if n_unk > 0:
            self.rev_n_unk  = 1./n_unk 
        else:
            self.rev_n_unk  = 1
        self.UNKw = spl_words
	#print >> sys.stderr, len(self.UNKw) 
        #Read ngram training examples and corresponding y labels 
        self.ntrain_set_x = NNLMdata[0][0]
        self.ntrain_set_y = NNLMdata[0][1]
        
        self.nvalid_set_x = NNLMdata[1][0]
        self.nvalid_set_y = NNLMdata[1][1]
        
        self.ntest_set_x =  NNLMdata[2][0]
        self.ntest_set_y =  NNLMdata[2][1]
        
        self.test_set_x_sparse=[]
        for test_set_xi in self.ntest_set_x:
	    if not UseDirect:
		if train_projection:
        	    shared_x = shared_data(convert_to_sparse(test_set_xi,N))
		else:
                    shared_x = shared_data(convert_to_projection(test_set_xi,self.pW,N))	
	    else:
                shared_x = shared_data(test_set_xi)
            self.test_set_x_sparse.append(shared_x) 

        self.valid_set_y  = shared_data(self.nvalid_set_y,"int")
        self.test_set_y   = shared_data(self.ntest_set_y,"int")

        if n_feats > 0:
            self.ntrain_set_featx = NNLMFeatData[0][0]
            self.ntrain_set_featy = NNLMFeatData[0][1]

            self.nvalid_set_featx = NNLMFeatData[1][0]
            self.nvalid_set_featy =  NNLMFeatData[1][1]
            
            self.ntest_set_featx =  NNLMFeatData[2][0]
            self.ntest_set_featy =  NNLMFeatData[2][1]
            # need to implement UseDirect / train projection  
            self.test_set_featx_sparse  = shared_data(convert_to_sparse_matrix_feature(self.ntest_set_featx,n_feats).todense())
            self.test_set_featy   = shared_data(self.ntest_set_featy,"int")
 	    
            self.train_set_featx_sparse_notshared  = convert_to_sparse_matrix_feature(self.ntrain_set_featx,n_feats) 

        if n_unk > 0:
            #valid_error_penalty = GetPenaltyVector(nvalid_set_y,rev_n_unk,UNKw)
            self.test_error_penalty  = GetPenaltyVector(self.ntest_set_y,self.rev_n_unk,self.UNKw)
            self.train_error_penalty_notshared = GetPenaltyVector(self.ntrain_set_y,self.rev_n_unk,self.UNKw,False)

        #convert training data to numpy arrays.
        self.train_set_x_sparse_notshared = []
        for train_set_xi in self.ntrain_set_x:
            if not UseDirect:
		if train_projection:
            	    train_x = convert_to_sparse_matrix(train_set_xi,N)
		else:
                    train_x = convert_to_projection(train_set_xi,self.pW,N)
	    else:
                train_x = train_set_xi

            self.train_set_x_sparse_notshared.append(train_x)
            
        ######################
        # BUILD ACTUAL MODEL # 
        ######################
        print >> sys.stderr, '...defining the model'

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch	
	if not UseDirect:
            self.x1 = T.matrix('x1')  # the data is presented as rasterized images
            self.x2 = T.matrix('x2')
            self.x3 = T.matrix('x3')
            self.x4 = T.matrix('x4')
            self.x5 = T.matrix('x5')
            self.x6 = T.matrix('x6')
            self.xfeat = T.matrix('xfeat') # if we hav features 
	else:
            self.x1 = T.vector('x1')  # the data is presented as rasterized images
            self.x2 = T.vector('x2')
            self.x3 = T.vector('x3')
            self.x4 = T.vector('x4')
            self.x5 = T.vector('x5')
            self.x6 = T.vector('x6')
            self.xfeat = T.vector('xfeat') # if we hav features
   
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        self.error_penalty = T.fvector('error_penalty') # In case some word are more important than others, 
	# we give them additional penalty. Leave as [] for uniform penalty
    
        rng = numpy.random.RandomState(1234)
        
        # construct the MLP class
        if ngram==2:
	    if shared==False:
                hWshared=hW; hBshared = hB;
            self.classifier = MLP(rng=rng, input=self.x1, nhistory = [] ,feature=self.xfeat,n_features=n_feats,n_in=1, n_P=P, n_hidden=H, 
				number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hWshared,hB=hBshared,lW=lW,lB=lB,train_projection=train_projection,new_weights=new_weights)
        elif ngram==3:
            # if this is not training, then we want hWshared,hBshared to be the loaded matrices 
	    if shared==False:
		hWshared=hW; hBshared = hB; 	  
            self.classifier = MLP(rng=rng, input=self.x1, nhistory = [self.x2] ,feature=self.xfeat,n_features=n_feats,n_in=1, n_P=P, n_hidden=H,
                                  number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hWshared,hB=hBshared,lW=lW,lB=lB,shared=shared,train_projection=train_projection,new_weights=new_weights)
        elif ngram==4:
            self.classifier = MLP(rng=rng, input=self.x1, nhistory = [self.x2,self.x3] ,feature=self.xfeat,n_features=n_feats,n_in=1, n_P=P, n_hidden=H, 
                                  number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
        elif ngram==5:
            self.classifier = MLP(rng=rng, input=self.x1, nhistory = [self.x2,self.x3,self.x4] ,feature=self.xfeat,n_features=n_feats,n_in=1, n_P=P, n_hidden=H, 
                                  number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
        elif ngram==6:
            self.classifier = MLP(rng=rng, input=self.x1, nhistory = [self.x2,self.x3,self.x4,self.x5] ,feature=self.xfeat,n_features=n_feats,n_in=1, n_P=P, 
                                  n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
        elif ngram==7:
            self.classifier = MLP(rng=rng, input=self.x1, nhistory = [self.x2,self.x3,self.x4,self.x5,self.x6] ,feature=self.xfeat,n_features=n_feats,n_in=1, 
                                  n_P=P, n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically

        '''if n_unk > 0:
	    print >> sys.stderr, "Error penalty enabled in optimization"
            self.cost = self.classifier.negative_log_likelihood(self.y,self.error_penalty) \
                + L1_reg * self.classifier.L1 \
                + L2_reg * self.classifier.L2_sqr
	

        else:
            self.cost = self.classifier.negative_log_likelihood(self.y) \
                + L1_reg * self.classifier.L1 \
                + L2_reg * self.classifier.L2_sqr
	'''
        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
    
        Tgivens = {}
        Tgivens[self.x1]= self.test_set_x_sparse[0][index * batch_size:(index + 1) * batch_size]
        if ngram>=3:
            Tgivens[self.x2] =  self.test_set_x_sparse[1][index * batch_size:(index + 1) * batch_size]
        if ngram>=4:
            Tgivens[self.x3] =  self.test_set_x_sparse[2][index * batch_size:(index + 1) * batch_size]
        if ngram>=5:
            Tgivens[self.x4] =  self.test_set_x_sparse[3][index * batch_size:(index + 1) * batch_size]
        if ngram>=6:
            Tgivens[self.x5] =  self.test_set_x_sparse[4][index * batch_size:(index + 1) * batch_size]
        if ngram>=7:
            Tgivens[self.x6] =  self.test_set_x_sparse[5][index * batch_size:(index + 1) * batch_size]
        if n_feats>0:
            Tgivens[self.xfeat] = self.test_set_featx_sparse[index * batch_size:(index + 1) * batch_size]

        if n_unk > 0:
            Tgivens[self.error_penalty] = self.test_error_penalty[index * batch_size:(index + 1) * batch_size]
            Touts = self.classifier.tot_ppl(self.y,self.error_penalty)
        else:
            Touts = self.classifier.tot_ppl(self.y)

        Tgivens[self.y] = self.test_set_y[index * batch_size:(index + 1) * batch_size]

        self.test_model = theano.function(inputs=[index],
                                          outputs= Touts,
                                          givens= Tgivens,
					  on_unused_input='warn')
    

        self.validate_model = theano.function(inputs=[index],
                                              outputs=Touts, 
                                              on_unused_input='warn',
                                              givens= Tgivens ) 
    
        param_out = [];
        param_out.append(self.classifier.get_params_pW())

    
        for i in range(number_hidden_layer):
            param_out.append(self.classifier.get_params_hW(i))
            param_out.append(self.classifier.get_params_hb(i))

        param_out.append(self.classifier.get_params_lW()); 
        param_out.append(self.classifier.get_params_lb()) ; 


        self.final_weights = theano.function(inputs=[], outputs= param_out) 
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        ''';self.gparams = []
        for param in self.classifier.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)
        '''
        # specify how to update the parameters of the model as a dictionary
        self.tot_train_size = len(self.ntrain_set_y)
        self.tot_valid_size = len(self.nvalid_set_y) 
	self.train_set_x_sparse =  None 

    def projection_cost(self,synset,mlp2,rev=False):
	cost = 0
	'''for w in synset:
	    if rev:
		w2 =w 
		w1 = synset[w]
	    else:
		w1 = w
		w2 = synset[w]
	    if cost==0:
		cost = ((self.classifier.projectionLayer.W[w1] - mlp2.classifier.projectionLayer.W[w2])**2).sum()
	    else:
		cost = cost + ((self.classifier.projectionLayer.W[w2] - mlp2.classifier.projectionLayer.W[w1])**2).sum()
          '''     
	print >> sys.stderr, "Cost is defined as W1 - W2 for ",synset,"nodes"  
        cost = ((self.classifier.projectionLayer.W[0:synset] - mlp2.classifier.projectionLayer.W[0:synset])**2).sum()
        cost = cost + ((self.classifier.logRegressionLayer.W[:,0:synset] - mlp2.classifier.logRegressionLayer.W[:,0:synset])**2).sum()
	return cost 

    def define_cost(self,synset,mlp2,rev=False):
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
		
        if self.n_unk > 0:
            print >> sys.stderr, "Error penalty enabled in optimization"
            self.cost = self.classifier.negative_log_likelihood(self.y,self.error_penalty) \
                + self.L1_reg * self.classifier.L1 \
                + self.L2_reg * self.classifier.L2_sqr


        else:
            self.cost = self.classifier.negative_log_likelihood(self.y) \
                + self.L1_reg * self.classifier.L1 \
                + self.L2_reg * self.classifier.L2_sqr


	if synset!=0: #{}:
	    print >> sys.stderr, "Structured projectionLayer regression"		
	    self.cost = self.cost + self.L2_struct * self.projection_cost(synset,mlp2,rev)
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        self.gparams = []
        for param in self.classifier.params:
            gparam = T.grad(self.cost, param)
            self.gparams.append(gparam)



    def define_train_model(self,parts_index,copy_size,batch_size,n_train_parts=10):
	if n_train_parts ==1 and self.counter>1:
	    return 1
        del self.train_set_x_sparse
        index = T.lscalar()    # index to a [mini]batch
	self.counter = self.counter + 1 
        self.train_set_x_sparse=[]
        for train_set_xi in self.train_set_x_sparse_notshared:
	    if not UseDirect:
		if self.train_projection:	
            	    tmp_matrix = train_set_xi[parts_index * copy_size:min(self.tot_train_size,(parts_index + 1) * copy_size)].todense()
		    shared_x = theano.shared( tmp_matrix,borrow=False )
		else:
                    tmp_matrix = train_set_xi[parts_index * copy_size:min(self.tot_train_size,(parts_index + 1) * copy_size)]
                    shared_x = theano.shared( tmp_matrix,borrow=False )
	    else:
                tmp_matrix = train_set_xi[parts_index * copy_size:min(self.tot_train_size,(parts_index + 1) * copy_size)]
	        shared_x = shared_data(tmp_matrix,sparse=False,borrow=False) #theano.shared( tmp_matrix,sparse=False,borrow=False )

            self.train_set_x_sparse.append(shared_x)

        Tgivens = {}
        Tgivens[self.x1] = self.train_set_x_sparse[0][index * batch_size:(index + 1) * batch_size]
        if self.ngram>=3:
            Tgivens[self.x2] =  self.train_set_x_sparse[1][index * batch_size:(index + 1) * batch_size]
        if self.ngram>=4:
            Tgivens[self.x3] =  self.train_set_x_sparse[2][index * batch_size:(index + 1) * batch_size]
        if self.ngram>=5:
            Tgivens[self.x4] =  self.train_set_x_sparse[3][index * batch_size:(index + 1) * batch_size]
        if self.ngram>=6:
            Tgivens[self.x5] =  self.train_set_x_sparse[4][index * batch_size:(index + 1) * batch_size]
        if self.ngram>=7:
            Tgivens[self.x6] =  self.train_set_x_sparse[5][index * batch_size:(index + 1) * batch_size]
        if self.n_feats>0:
	    # need to implement projection 
     	    tmp_matrix     = self.train_set_featx_sparse_notshared[parts_index * copy_size:min(self.tot_train_size,(parts_index + 1) * copy_size) ]
	    tmp_matrix = tmp_matrix.todense()
            shared_featx   = theano.shared(tmp_matrix,borrow=False)
            Tgivens[self.xfeat] = shared_featx[index * batch_size:(index + 1) * batch_size]

        self.train_set_y = shared_data(self.ntrain_set_y[parts_index * copy_size :min(self.tot_train_size,(parts_index + 1)* copy_size) ],"int")
        Tgivens[self.y] = self.train_set_y[index * batch_size:(index + 1) * batch_size] 

        if self.n_unk > 0:
            self.train_error_penalty = theano.shared(self.train_error_penalty_notshared[parts_index * copy_size:min(self.tot_train_size,(parts_index + 1) * copy_size) ],borrow=False) 

            Tgivens[self.error_penalty] = self.train_error_penalty[index * batch_size:(index + 1) * batch_size]
        self.train_model = theano.function(inputs=[index], outputs=self.cost,
                                      updates=self.updates,
				      on_unused_input='ignore',
                                      givens= Tgivens)
    def delete_shared(self):
        del self.train_set_x_sparse
    def define_updates(self,learning_rate):
        self.updates = []
        self.learning_rate = learning_rate 
        for param, gparam in zip(self.classifier.params, self.gparams):
            self.updates.append(( param, param - self.learning_rate * gparam))


        
def train_multi_mlp(NNLMdata_list,NNLMFeatData_list,OldParams_list,ngram_list,n_feats_list,n_unk_list,N_list,P,H,number_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam_list,gpu_copy_size_list,spl_words_list, train_projection_list,new_weights_list,synset, number_of_languages):

    copy_size = min(gpu_copy_size_list )/ len(gpu_copy_size_list)
    learning_rate0 = learning_rate
    print >> sys.stderr, "building models..."
    Trainers=[]
    for i in range(number_of_languages):
	if Trainers == [] :
            Train1 = TrainMLP(NNLMdata_list[i],NNLMFeatData_list[i],OldParams_list[i],ngram_list[i],n_feats_list[i],n_unk_list[i],N_list[i],P,H,
                              number_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam_list[i],copy_size,spl_words_list[i],train_projection_list[i],new_weights_list[i])
	else:
            Train1 = TrainMLP(NNLMdata_list[i],NNLMFeatData_list[i],OldParams_list[i],ngram_list[i],n_feats_list[i],n_unk_list[i],N_list[i],P,H,
                              number_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam_list[i],copy_size,
                              spl_words_list[i],train_projection_list[i],new_weights_list[i],Trainers[0].classifier.hsharedW,Trainers[0].classifier.hsharedb,True)

        Trainers.append(Train1) 

    if synset!=0: #{}:
	Trainers[0].define_cost(synset,Trainers[1],False)
	
	Trainers[1].define_cost(synset,Trainers[0],True)
    else:
    	for mlp in Trainers:
	    mlp.define_cost(synset,mlp,False)

    print >> sys.stderr, '... training'
    tot_train_size = max(Trainers[i].tot_train_size for i in range(number_of_languages)) 
    tot_valid_size = min(Trainers[i].tot_valid_size for i in range(number_of_languages))

    # Get counts etc...
    n_train_parts = tot_train_size/(copy_size) +  int( tot_train_size%copy_size!=0)
    n_train_batches = copy_size/batch_size
    batch_size_train = batch_size
    n_valid_batches = tot_valid_size / batch_size
    n_test_batches = tot_valid_size / batch_size


    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.998  # a relative improvement of this much is
                                   # considered significant
    epsilon = (1-improvement_threshold) *0.05; #parameter for adaptive learning rate 
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    # Initialize training 
    best_params = None
    best_validation_loss = {}
    for l in range(number_of_languages):
    	best_validation_loss[l] = numpy.inf
    best_iter = 0.
    test_score = 0.
    start_time = time.clock()
    epoch = 0
    done_looping = False    
    
    print >> sys.stderr, "Training parts:", n_train_parts,",Copy size:",copy_size,",Total training size:",tot_train_size
    params = {}
    for l in range(number_of_languages):
    	params[l] = Trainers[l].final_weights()

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        totcost=0
	repeater=100
	repeat_counter = 2
        for i in range(number_of_languages):
            Trainers[i].define_updates(learning_rate)

        print >> sys.stderr, "Epoch:",epoch,", Current learning rate:", learning_rate
	
        for parts_index in range(n_train_parts):

            n_train_batches = ( min(tot_train_size, (parts_index+ 1) * copy_size) - parts_index * copy_size  )/batch_size
            #for i in range(number_of_languages):
	    #		Trainers[i].delete_shared()
	    for i in range(number_of_languages):
                if Trainers[i].tot_train_size > ( parts_index * copy_size ) : # and ( epoch==1 or n_train_parts>1) :	
            	    Trainers[i].define_train_model(parts_index,copy_size,batch_size,n_train_parts)
		else:
		    repeater = repeater + 1 
		    if Trainers[i].tot_train_size <  repeater * copy_size:
			repeat_counter = repeat_counter + 1 
			print >> sys.stderr, "repeater",repeater,"New learning rate for language",i,":",learning_rate/repeat_counter 
	            	Trainers[i].define_updates(learning_rate/repeat_counter)
			repeater = 0 

                    Trainers[i].define_train_model(repeater,copy_size,batch_size)
		  
 		
            print >> sys.stderr, "Training part:", parts_index+1 ,"of",n_train_parts
            print >> sys.stderr,"shuffling data... ",
	    xn = numpy.arange(n_train_batches)
            numpy.random.shuffle(xn)
	    print >> sys.stderr,"shuffled"
            for minibatch_index in xn: #xrange(n_train_batches):
		validation_frequency = min(n_train_batches, patience / 2)
                if minibatch_index * batch_size > tot_train_size:
                    break

                for i in range(number_of_languages):
	            if Trainers[i].tot_train_size > ( parts_index * copy_size  + minibatch_index * batch_size)  :
                    	minibatch_avg_cost = Trainers[i].train_model(minibatch_index)
                    	totcost+=minibatch_avg_cost
		    elif Trainers[i].tot_train_size <= ( parts_index * copy_size ) and Trainers[i].tot_train_size >  repeater * copy_size + minibatch_index * batch_size :
                        minibatch_avg_cost = Trainers[i].train_model(minibatch_index)
                        totcost+=minibatch_avg_cost
                
                 
                iter = epoch * n_train_batches + minibatch_index
                
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
		    for l in range(number_of_languages):
                    	validation_losses = [Trainers[l].validate_model(i) for i
                        	                 in xrange(n_valid_batches)]
                    	this_validation_loss = numpy.exp(numpy.mean(validation_losses)) 
                     
                    	print >> sys.stderr, ('Language %i, minibatch %i/%i, validation error %f %%' %
                          	(l, minibatch_index + 1, n_train_batches,
                           	this_validation_loss))

                    # if we got the best validation score until now
                  	if this_validation_loss < best_validation_loss[l]:
                    #improve patience if loss improvement is good enough
			    done_looping= False;
                            if this_validation_loss < best_validation_loss[l] *  \
                                    improvement_threshold:
                            	patience = max(patience, iter * patience_increase)

                            best_validation_loss[l] = this_validation_loss
                            best_iter = iter

			    test_score = this_validation_loss
			
			    params[l] = Trainers[l].final_weights()
        		    write_all_params_matlab(fparam_list[l],params[l],number_hidden_layer)
                            print >> sys.stderr, ((' Language %i, epoch %i, minibatch %i/%i, valid error of '
                             	'best model %f %%') %
                              	(l, epoch, minibatch_index + 1, n_train_batches,
                               	test_score))
			
            gc.collect()
        if adaptive_learning_rate:
	    if learning_rate > 0.004:
	        learning_rate = float(learning_rate0)/(1 + float(iter*epsilon))
            else:
		learning_rate = 0.005
	
        fl = '/tmp/stoppage'
        stop=0
        for sl in open(fl,'r'):
            if sl.strip()=="STOP":
                stop=1
        if stop==1:
            break
        if patience <= iter:
            done_looping = True
            print >> sys.stderr,  "done looping","patience:",patience,"iter:",iter   
	    break
        print >> sys.stderr, "Total Cost in traning:", totcost
    
    end_time = time.clock()
    for l in range(number_of_languages):
    	print >> sys.stderr, (('Optimization complete. Best validation score of %f %% '
        	   'obtained at iteration %i, with test performance %f %%') %
          	(best_validation_loss[l], best_iter, test_score ))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    for l in range(number_of_languages):
    	write_all_params_matlab(fparam_list[l],params[l],number_hidden_layer)


def test_mlp(testfile,testfeat,paramdir,outfile="", batch_size=50,write_arpa = False):
    copy_size = 20000
    if outfile == "":
	outfile = testfile+".prob" 
    ngram,n_feats,N,P,H,number_hidden_layer,WordID = read_machine(paramdir) 
    unkid = -1
    if "<UNK>" in WordID:
	unkid = WordID["<UNK>"]
    if write_arpa == False:
        TestData,N_input_layer,N_unk = CreateData(testfile,WordID,[],ngram,False,False) 
        NNLMdata = load_data(TestData,ngram,N,1e10,unkid)
    else:
        NNLMdata = load_data_from_file(testfile,ngram,N,1e10,unkid)

    if n_feats > 0:
	if testfeat=="None":
	     print >> sys.stderr, "number of feats", n_feats," but feature file not provided. Exiting" 
	     sys.exit(1)
	unkid = -1 
	NNLMFeatData = load_data_from_file(testfeat,ngram,N*1000,unkid) 
        ntest_set_featx =  NNLMFeatData[0]
        ntest_set_featy =  NNLMFeatData[1]
        test_set_featx_sparse  = convert_to_sparse_matrix_feature(ntest_set_featx,n_feats).todense()
        
    pW,hW,hB,lW,lB = load_params_matlab_multi(paramdir,number_hidden_layer)
    print number_hidden_layer
    ntest_set_x =  NNLMdata[0]
    ntest_set_y =  NNLMdata[1]

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    input_symbols = [] 
    if not UseDirect:
    	x1 = T.matrix('x1') ; input_symbols.append(x1) # the data is presented as rasterized images
    	x2 = T.matrix('x2') ; input_symbols.append(x2)
	x3 = T.matrix('x3') ; input_symbols.append(x3)
    	x4 = T.matrix('x4') ; input_symbols.append(x4)
    	x5 = T.matrix('x5') ; input_symbols.append(x5)
    	x6 = T.matrix('x6') ; input_symbols.append(x6)
    	xfeat = T.matrix('xfeat') # if we hav features
    else:
        x1 = T.vector('x1') ; input_symbols.append(x1) # the data is presented as rasterized images
        x2 = T.vector('x2') ; input_symbols.append(x2)
        x3 = T.vector('x3') ; input_symbols.append(x3)
        x4 = T.vector('x4') ; input_symbols.append(x4)
        x5 = T.vector('x5') ; input_symbols.append(x5)
        x6 = T.vector('x6') ; input_symbols.append(x6)
        xfeat = T.vector('xfeat') # if we hav features

    y = T.ivector('y')  # the labels are presented as 1D vector of
    error_penalty = T.fvector('error_penalty') # In case some word are more important than others, we give them additional penalty. Leave as [] for uniform penalty


    rng = numpy.random.RandomState(1234)
    # construct the MLP class

    inputs = [];nhistory=[];
    if write_arpa == False:
	inputs.append(y)
    inputs.append(xfeat); inputs.append(x1)
    for i in range(1,ngram-1):
	inputs.append(input_symbols[i])
	nhistory.append(input_symbols[i])

    classifier = MLP(rng=rng, input=x1, nhistory = nhistory,feature=xfeat,n_features=n_feats,n_in=1, n_P=P, n_hidden=H, 
                     number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
 
    outputs = []
    if write_arpa == False:
	outputs.append(classifier.get_p_y_given_x(y))
	outputs.append(classifier.tot_ppl(y))
    else:
	outputs.append(classifier.get_p_y_given_x_all())
    
    probs_model_full = theano.function(inputs= inputs, outputs= outputs, on_unused_input = 'warn') 


    foutPF= open(outfile,'w')
    foutPFUnk = open(outfile+'.unk','w')
    gc.collect()
    t = len(ntest_set_y)
    if copy_size > t:
        test_batches=1

    else:
        test_batches = t/copy_size + 1
    test_ppls = []
    for i in range(test_batches):
	print >> sys.stderr,"running batch", i+1,"of ",test_batches 
	x_sparse=[]
	for set_x in ntest_set_x:
	    if not UseDirect:
            	x = convert_to_sparse(set_x[i*copy_size: (i+1)*copy_size],N)
	    else:
		x = numpy.asarray(set_x[i*copy_size: (i+1)*copy_size],dtype=theano.config.floatX)

  	    x_sparse.append(x)
        y_test = ntest_set_y[i*copy_size : (i+1)*copy_size] 
	if n_feats > 0:
 	    x_feat = test_set_featx_sparse[i * copy_size:(i + 1) * copy_size]
	else:
	    if not UseDirect:
	    	x_feat = [[],[]]
	    else:
		x_feat = []
	#print y_test, x_feat, x_sparse[0], x_sparse[1]
	if write_arpa == False:
      	    scores,loss=  probs_model_full(y_test,x_feat,*x_sparse) 
	    
	    test_ppls.append(loss)
	
            for yl,y in zip(scores,y_test):
	       	if y==WordID['<UNK>']:
		    print >> foutPF, 0
		    print >> foutPFUnk, numpy.exp(yl)
		    continue
            	print >> foutPF, numpy.exp(yl)
		print >> foutPFUnk , numpy.exp(yl)
	else:
	    scores = probs_model_full(x_feat,*x_sparse) 
 	    for yl in scores:
            	for xi in yl:
                    i=0;
                    for pi in xi:
                    	print >> foutPF, numpy.exp(pi)
	
        gc.collect()
    ppl = numpy.exp( numpy.mean(test_ppls))
    print "Perplexity with OOVs:",ppl 


if __name__ == '__main__':
    test_mlp()

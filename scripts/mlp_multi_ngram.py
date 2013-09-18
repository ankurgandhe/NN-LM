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

copy_size=50000

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
        lin_output = T.dot(self.input, self.W)
        for history in nhistory:
            lin_output = T.concatenate((lin_output,T.dot(history,self.W)),axis=1)
         
        if n_feat==0:
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
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activation
    """
    
    def __init__(self, rng, input, nhistory, feature,n_features,n_languages , n_in,n_P, n_hidden,number_hidden_layer, n_out, ngram=3, Voc=4096,pW=None,hW=None,hB=None,lW=None,lB=None):
        self.projectionLayers=[]
	for i in range(n_languages):
	    projectionLayer = ProjectionLayer_ngram(rng=rng, input=input, nhistory=nhistory, feature=feature, n_feat = n_features, n_in=1, n_out=n_P, W=pW, N=Voc[i],activation=None)
	    self.projectionLayers.append(projectionLayer)
	
	self.HiddenLayers=[]
	self.nH = number_hidden_layer
	for i in range(number_hidden_layer): 
	    if hW==None:
		hweights = None
		hbias = None 
	    else:
		hweights = hW[i]
		hbias = hB[i]
	    if i==0:
             	hiddenLayer = HiddenLayer(rng=rng, input=self.projectionLayers[0].output,
             	                          n_in=n_P*(ngram-1)+n_features, n_out=n_hidden,
                	                       activation=T.tanh,W=hweights,b=hbias)
	    else:
		hiddenLayer = HiddenLayer(rng=rng, input=self.HiddenLayers[i-1].output,
                                          n_in=n_hidden, n_out=n_hidden,
                                               activation=T.tanh,W=hweights,b=hbias)	
	    self.HiddenLayers.append(hiddenLayer)
        # The logistic regression layer gets as input the hidden units++++++++++++++++++++
        # of the hidden layer
	self.logRegressionLayers=[]
	for i in range(n_languages):
            logRegressionLayer = LogisticRegression(input=self.HiddenLayers[number_hidden_layer-1].output,
                                                     n_in=n_hidden,
                                                     n_out=n_out[i],W=lW,b=lB)
	    self.logRegressionLayers.append(logRegressionLayer)
	
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
	self.L1 = 0
	for i in range(n_languages):
	    self.L1 = self.L1 + abs(self.logRegressionLayers[i].W).sum()
	for i in range(number_hidden_layer):
	    self.L1  = self.L1 + abs(self.HiddenLayers[i].W).sum()
        #self.L1 = abs(self.hiddenLayer.W).sum() \
        #        + abs(self.logRegressionLayer.W).sum() 


        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
	self.L2_sqr = 0
	for i in range(n_languages):
	    self.L2_sqr = self.L2_sqr + (self.logRegressionLayers[i].W ** 2).sum() 
        for i in range(number_hidden_layer):
	    self.L2_sqr = self.L2_sqr + (self.HiddenLayers[i].W ** 2).sum()
        #self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
        #            + (self.logRegressionLayer.W ** 2).sum() \


        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        #self.negative_log_likelihood = 0
	#for i in range(n_languages):
	#    self.negative_log_likelihood = self.negative_log_likelihood + self.logRegressionLayers[i].negative_log_likelihood
        # same holds for the function computing the number of errors
	#self.tot_ppl = 0
	#for i in range(n_languages):
	#     self.tot_ppl  = self.tot_ppl + self.logRegressionLayers[i].tot_ppl

        #self.get_p_y_given_x = self.logRegressionLayer.get_p_y_given_x2
        # the parameters of the model are the parameters of the two layer it is
        # made out o
        
        self.params = []; #self.projectionLayer.params #+ #self.hiddenLayer.params + self.logRegressionLayer.params  
	for i in range(number_hidden_layer):
	    self.params = self.params + self.HiddenLayers[i].params 
            self.params = self.params + self.logRegressionLayers[i].params

        
    def negative_log_likelihood(self,y,lang,penalty=[]):
        return self.logRegressionLayers[lang].negative_log_likelihood(y,penalty)
    def tot_ppl (self, y,lang, penalty=[]) : 
        return self.logRegressionLayers[lang].tot_ppl(y,penalty)
    def get_p_y_given_x(self,y,lang,penalty):
        return self.logRegressionLayers[lang].get_p_y_given_x2(y,penalty) 

    def get_params_pW(self,i):
        return (self.projectionLayers[i].W)
    def get_params_hW(self,i):
        return self.HiddenLayers[i].W #for i in range(self.nH)]#self.hiddenLayer.W
    def get_params_hb(self,i):
        return self.HiddenLayers[i].b #for i in range(self.nH)]#self.hiddenLayer.b
    def get_params_lW(self,i):
        return self.logRegressionLayers[i].W 
    def get_params_lb(self,i):
        return self.logRegressionLayers[i].b

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

def GetPenaltyVector(y,penalty,Wids):
    # Wids = list of words that need to be penalized with penalty
    vec_penalty  = zeros(len(y ))
    i  = 0
    count = 0
    while i < len(vec_penalty):
        if y[i] in Wids: 
            vec_penalty[i]=penalty
	    count = count + 1
        else:
            vec_penalty[i]=1
        i=i+1
        
        vec_penalty_shared  =  theano.shared(numpy.asarray(vec_penalty,
                                                           dtype=theano.config.floatX),
                             borrow=True)
    print >> sys.stderr, "Penalized ",count," words", "with penalty", penalty 
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

def train_multi_mlp(NNLMdata,NNLMFeatData,OldParams,ngram,n_feats,n_unk,N,P,H,number_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,gpu_copy_size,n_languages):
            
    copy_size = gpu_copy_size 
    learning_rate0 = learning_rate
    if n_unk > 0:
	rev_n_unk  = 1./n_unk 
    else:
	rev_n_unk  = 1
    UNKw = []
    #Read ngram training examples and corresponding y labels 
    ntrain_set_x = NNLMdata[0][0]
    ntrain_set_y = NNLMdata[0][1]
        
    nvalid_set_x = NNLMdata[1][0]
    nvalid_set_y = NNLMdata[1][1]
    
    ntest_set_x =  NNLMdata[2][0]
    ntest_set_y =  NNLMdata[2][1]
    
    #Convert valid and test set to sparse and shared objects 
    #valid_set_x_sparse=[]
    #for valid_set_xi in nvalid_set_x:
        #shared_x = shared_data(convert_to_sparse(valid_set_xi,N))
	#shared_x = convert_to_sparse(valid_set_xi,N)
        #valid_set_x_sparse.append(shared_x) 

    test_set_x_sparse=[]
    for test_set_xi in ntest_set_x:
	shared_x = shared_data(convert_to_sparse(test_set_xi,N[0]))
        test_set_x_sparse.append(shared_x) 

    valid_set_y  = shared_data(nvalid_set_y,"int")
    test_set_y   = shared_data(ntest_set_y,"int")
    test_set_lang = shared_data(numpy.ones_like(ntest_set_y),"int")

    if n_feats > 0:
        ntrain_set_featx = NNLMFeatData[0][0]
        ntrain_set_featy = NNLMFeatData[0][1]

        nvalid_set_featx = NNLMFeatData[1][0]
        nvalid_set_featy =  NNLMFeatData[1][1]

        ntest_set_featx =  NNLMFeatData[2][0]
        ntest_set_featy =  NNLMFeatData[2][1]

        test_set_featx_sparse  = shared_data(convert_to_sparse_matrix_combine(ntest_set_featx,n_feats).todense())

        test_set_featy   = shared_data(ntest_set_featy,"int")
        
    #UNKw.append(2)
    if n_unk > 0:
        #valid_error_penalty = GetPenaltyVector(nvalid_set_y,rev_n_unk,UNKw)
        test_error_penalty  = GetPenaltyVector(ntest_set_y,rev_n_unk,UNKw)
            
    ######################
    # BUILD ACTUAL MODEL # 
    ######################
    print >> sys.stderr, '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x1 = T.matrix('x1')  # the data is presented as rasterized images
    x2 = T.matrix('x2')
    x3 = T.matrix('x3')
    x4 = T.matrix('x4')
    x5 = T.matrix('x5')
    x6 = T.matrix('x6')
    xfeat = T.matrix('xfeat') # if we hav features 
    y = T.ivector('y')  # the labels are presented as 1D vector of
    lang = T.ivector('lang') # The language of each training example 
    error_penalty = T.fvector('error_penalty') # In case some word are more important than others, we give them additional penalty. Leave as [] for uniform penalty
    
    rng = numpy.random.RandomState(1234)
    n_langs = 1 
    # construct the MLP class
    if OldParams:
        pW,hW,hB,lB,lW = OldParams
	learning_rate = 0.05
    else:
        pW = hW = hB = lB =  lW =  None
    if ngram==2:
        classifier = MLP(rng=rng, input=x1, nhistory = [] ,feature=xfeat,n_features=n_feats,n_languages=n_langs,n_in=1, n_P=P, n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
    elif ngram==3:
        classifier = MLP(rng=rng, input=x1, nhistory = [x2] ,feature=xfeat,n_features=n_feats,n_languages=n_langs,n_in=1, n_P=P, n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
    elif ngram==4:
        classifier = MLP(rng=rng, input=x1, nhistory = [x2,x3] ,feature=xfeat,n_features=n_feats,n_languages=n_langs,n_in=1, n_P=P, n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
    elif ngram==5:
        classifier = MLP(rng=rng, input=x1, nhistory = [x2,x3,x4] ,feature=xfeat,n_features=n_feats,n_languages=n_langs,n_in=1, n_P=P, n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
    elif ngram==6:
        classifier = MLP(rng=rng, input=x1, nhistory = [x2,x3,x4,x5] ,feature=xfeat,n_features=n_feats,n_languages=n_langs,n_in=1, n_P=P, n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
    elif ngram==7:
        classifier = MLP(rng=rng, input=x1, nhistory = [x2,x3,x4,x5,x6] ,feature=xfeat,n_features=n_feats,n_languages=n_langs,n_in=1, n_P=P, n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    if n_unk > 0:
        cost = classifier.negative_log_likelihood(y,lang,error_penalty) \
            + L1_reg * classifier.L1 \
            + L2_reg * classifier.L2_sqr
    else:
        cost = classifier.negative_log_likelihood(y,lang) \
            + L1_reg * classifier.L1 \
            + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    
    Tgivens = {}
    Tgivens[x1]= test_set_x_sparse[0][index * batch_size:(index + 1) * batch_size]
    if ngram>=3:
        Tgivens[x2] =  test_set_x_sparse[1][index * batch_size:(index + 1) * batch_size]
    if ngram>=4:
        Tgivens[x3] =  test_set_x_sparse[2][index * batch_size:(index + 1) * batch_size]
    if ngram>=5:
        Tgivens[x4] =  test_set_x_sparse[3][index * batch_size:(index + 1) * batch_size]
    if ngram>=6:
        Tgivens[x5] =  test_set_x_sparse[4][index * batch_size:(index + 1) * batch_size]
    if ngram>=7:
        Tgivens[x6] =  test_set_x_sparse[5][index * batch_size:(index + 1) * batch_size]
    if n_feats>0:
        Tgivens[xfeat] = test_set_featx_sparse_notshared[index * batch_size:(index + 1) * batch_size]

    if n_unk > 0:
        Tgivens[error_penalty] = test_error_penalty[index * batch_size:(index + 1) * batch_size]
        Touts = classifier.tot_ppl(y,lang,error_penalty)
    else:
        Touts = classifier.tot_ppl(y,lang)

    Tgivens[y] = test_set_y[index * batch_size:(index + 1) * batch_size]
    Tgivens[lang] = test_set_lang[index * batch_size:(index + 1) * batch_size]
        
    test_model = theano.function(inputs=[index],
                                 outputs= Touts,
                                 on_unused_input='warn',
                                 givens= Tgivens )

    validate_model = theano.function(inputs=[index],
                                     outputs=Touts, 
                                     on_unused_input='warn',
                                     givens= Tgivens ) 

    param_out = [];
    param_out.append(classifier.get_params_pW())
    for i in range(number_hidden_layer):
	param_out.append(classifier.get_params_hW(i))
	param_out.append(classifier.get_params_hb(i))
    param_out.append(classifier.get_params_lW()); 
    param_out.append(classifier.get_params_lb()) ; 

    final_weights = theano.function(inputs=[], outputs= param_out) 
    # Get counts etc... 
    tot_train_size = len(ntrain_set_y)    
    n_train_parts = tot_train_size/(copy_size) +  int( tot_train_size%copy_size!=0)
    n_train_batches = copy_size/batch_size
    batch_size_train = batch_size
    n_valid_batches = len(nvalid_set_y) / batch_size
    n_test_batches = len(ntest_set_y)/ batch_size   

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)
        
    # specify how to update the parameters of the model as a dictionary
    updates = {}
    for param, gparam in zip(classifier.params, gparams):
        updates[param] = param - learning_rate * gparam

    print >> sys.stderr, '... training'

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
    best_validation_loss = numpy.inf
    best_iter = 0.
    test_score = 0.
    start_time = time.clock()
    epoch = 0
    done_looping = False    
    
    print >> sys.stderr, "Training parts:", n_train_parts,",Copy size:",copy_size,",Total training size:",tot_train_size
    #pW,hW,hB,lW,lB = final_weights() 
    params = final_weights()
    #share training y , training features 
    #train_set_y = shared_data(ntrain_set_y,"int")
    if n_feats>0:
        train_set_featx_sparse_notshared  = convert_to_sparse_matrix_combine(ntrain_set_featx,n_feats)
 
    #convert training data to numpy arrays.
    train_set_x_sparse_notshared = [] 
    for train_set_xi in ntrain_set_x:
        train_x = convert_to_sparse_matrix(train_set_xi,N[0])
        train_set_x_sparse_notshared.append(train_x)

    #train_set_y = shared_data(ntrain_set_y,"int")
    ntrain_set_lang =  ones_like(ntrain_set_y)

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        totcost=0
	updates = {}
	for param, gparam in zip(classifier.params, gparams):
            updates[param] = param - learning_rate * gparam

        print >> sys.stderr, "Current learning rate:", learning_rate
	#train_set_y = shared_data(ntrain_set_y,"int")
        for parts_index in range(n_train_parts):

            #convert training integers  1-of-N vectors into shared objects 
            train_set_x_sparse=[]
            for train_set_xi in train_set_x_sparse_notshared:
		#print >> sys.stderr , "Converting to dense...",
		tmp_matrix = train_set_xi[parts_index * copy_size:min(tot_train_size,(parts_index + 1) * copy_size)].todense()
		#print >> sys.stderr , "Conversion Done "
                shared_x = theano.shared( tmp_matrix, #train_set_xi[parts_index * copy_size:min(tot_train_size,(parts_index + 1) * copy_size)].todense(),
                                          borrow=False )
                train_set_x_sparse.append(shared_x)
           
            Tgivens = {}
            Tgivens[x1] = train_set_x_sparse[0][index * batch_size:(index + 1) * batch_size]
            if ngram>=3:
                Tgivens[x2] =  train_set_x_sparse[1][index * batch_size:(index + 1) * batch_size]
            if ngram>=4:
                Tgivens[x3] =  train_set_x_sparse[2][index * batch_size:(index + 1) * batch_size]
            if ngram>=5:
                Tgivens[x4] =  train_set_x_sparse[3][index * batch_size:(index + 1) * batch_size]
            if ngram>=6:
                Tgivens[x5] =  train_set_x_sparse[4][index * batch_size:(index + 1) * batch_size]
            if ngram>=7:
                Tgivens[x6] =  train_set_x_sparse[5][index * batch_size:(index + 1) * batch_size]
            if n_feats>0:
                Tgivens[xfeat] = theano_shared(train_set_featx_sparse_notshared[parts_index * copy_size + index * batch_size:parts_index * copy_size + (index + 1) * batch_size].todense(),borrow=False)
	    
	    
	    train_set_y = shared_data(ntrain_set_y[parts_index * copy_size :min(tot_train_size,(parts_index + 1)* copy_size) ],"int")
            train_set_lang = shared_data(ntrain_set_lang[parts_index * copy_size :min(tot_train_size,(parts_index + 1)* copy_size) ],"int")

            Tgivens[y] = train_set_y[index * batch_size:(index + 1) * batch_size] #[parts_index * copy_size +  index * batch_size:parts_index * copy_size +  (index + 1) * batch_size]
            Tgivens[lang] = train_set_lang[index * batch_size:(index + 1) * batch_size] 

            if n_unk > 0:
                train_error_penalty = GetPenaltyVector(ntrain_set_y[parts_index * copy_size:min(tot_train_size, (parts_index+ 1) * copy_size)],rev_n_unk,UNKw)
                Tgivens[error_penalty] = train_error_penalty[index * batch_size:(index + 1) * batch_size]
            
	    #print >> sys.stderr,"defining function for part ", parts_index+1,"..",
            train_model = theano.function(inputs=[index], outputs=cost,                                                                      
                                          updates=updates,                     
                                          givens= Tgivens)
            #print >> sys.stderr,"defined" 
            n_train_batches = len(ntrain_set_y[parts_index * copy_size:min(tot_train_size, (parts_index+ 1) * copy_size)])/batch_size
           
            print >> sys.stderr, "Training part:", parts_index+1 ,"of",n_train_parts
            for minibatch_index in xrange(n_train_batches):
		validation_frequency = min(n_train_batches, patience / 2)
                if minibatch_index * batch_size > len(ntrain_set_y):
                    break
                minibatch_avg_cost = train_model(minibatch_index)
                totcost+=minibatch_avg_cost
                
                iter = epoch * n_train_batches + minibatch_index
                
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.exp(numpy.mean(validation_losses)) #numpy.power(10, numpy.mean(validation_losses))
                     
                    print >> sys.stderr, ('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
			done_looping= False;
                        if this_validation_loss < best_validation_loss *  \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        #test_losses = [test_model(i) for i
                        #               in xrange(n_test_batches)]
                        #test_score = numpy.exp(numpy.mean(test_losses))
			test_score = this_validation_loss
			#pW,hW,hB,lW,lB = final_weights()
			params = final_weights()
                        print >> sys.stderr, (('     epoch %i, minibatch %i/%i, valid error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score))

            gc.collect()
            del train_set_x_sparse
            
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
    print >> sys.stderr, (('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss, best_iter, test_score ))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    #write_params_matlab(fparam,pW,hW,hB,lB,lW)
    write_all_params_matlab(fparam,params,number_hidden_layer)


def test_multi_mlp(testfile,testfeat,paramdir,outfile="", batch_size=50,write_arpa = False):
    copy_size = 20000
    if outfile == "":
	outfile = testfile+".prob" 
    ngram,n_feats,N,P,H,number_hidden_layer,WordID = read_machine(paramdir) 

    if write_arpa == False:
        TestData,N_input_layer,N_unk = CreateData(testfile,WordID,[],ngram,False,False) 
        NNLMdata = load_data(TestData,ngram,N)
    else:
        NNLMdata = load_data_from_file(testfile,ngram,N,1e10)

    if n_feats > 0:
	if testfeat=="None":
	     print >> sys.stderr, "number of feats", n_feats," but feature file not provided. Exiting" 
	     sys.exit(1)
	NNLMFeatData = load_data_from_file(testfeat,ngram,N) 
        ntest_set_featx =  NNLMFeatData[0]
        ntest_set_featy =  NNLMFeatData[1]
        test_set_featx_sparse  = convert_to_sparse_combine(ntest_set_featx,n_feats)
        #test_set_featy   = shared_data(ntest_set_featy,"int")

    pW,hW,hB,lW,lB = load_params_matlab_multi(paramdir,number_hidden_layer)
    print number_hidden_layer
    ntest_set_x =  NNLMdata[0]
    ntest_set_y =  NNLMdata[1]

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    input_symbols = [] 
    x1 = T.matrix('x1') ; input_symbols.append(x1) # the data is presented as rasterized images
    x2 = T.matrix('x2') ; input_symbols.append(x2)
    x3 = T.matrix('x3') ; input_symbols.append(x3)
    x4 = T.matrix('x4') ; input_symbols.append(x4)
    x5 = T.matrix('x5') ; input_symbols.append(x5)
    x6 = T.matrix('x6') ; input_symbols.append(x6)
    xfeat = T.matrix('xfeat') # if we hav features
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

    classifier = MLP(rng=rng, input=x1, nhistory = nhistory,feature=xfeat,n_features=n_feats,n_languages=n_langs,n_in=1, n_P=P, n_hidden=H, number_hidden_layer = number_hidden_layer, n_out=N, Voc=N,ngram=ngram,pW=pW,hW=hW,hB=hB,lW=lW,lB=lB,)
 
    outputs = []
    if write_arpa == False:
	outputs.append(classifier.get_p_y_given_x(y))
	outputs.append(classifier.tot_ppl(y))
    else:
	outputs.append(classifier.get_p_y_given_x_all())
    
    probs_model_full = theano.function(inputs= inputs, outputs= outputs, on_unused_input = 'warn') 


    foutPF= open(outfile,'w')
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
            x = convert_to_sparse(set_x[i*copy_size: (i+1)*copy_size],N)
            x_sparse.append(x)
        y_test = ntest_set_y[i*copy_size : (i+1)*copy_size]
        if n_feats > 0:
            x_feat = test_set_featx_sparse[i * copy_size:(i + 1) * copy_size]
        else:
            x_feat = [[],[]]
        #print y_test, x_feat, x_sparse[0], x_sparse[1]
        if write_arpa == False:
            scores,loss=  probs_model_full(y_test,x_feat,*x_sparse)

            test_ppls.append(loss)

            for yl,y in zip(scores,y_test):
                if y==WordID['<UNK>']:
                    print >> foutPF, 0
                    continue
                print >> foutPF, numpy.exp(yl)
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

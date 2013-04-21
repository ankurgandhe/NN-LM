import sys 
from ReadConfig  import * 
from Corpus import CreateData 
from NNLMio import *
from mlp_trigram import train_mlp


def print_params(foldparam,ngram,N_input_layer,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam):
    print >> sys.stderr, "system parameters:"
    print >> sys.stderr, "Vocab size:", N_input_layer
    print >> sys.stderr, "Projection layer:", P_projection_layer
    print >> sys.stderr, "Hidden layer:", H_hidden_layer
    print >> sys.stderr,  "learning rate:", learning_rate, "adaptive:", adaptive_learning_rate
    print >> sys.stderr, "L1_reg:", L1_reg
    print >> sys.stderr, "L2_reg:",L2_reg
    print >> sys.stderr, "max epochs:", n_epochs
    print >> sys.stderr, "batch_size:", batch_size
    print >> sys.stderr, "old params dir:", foldparam
    print >> sys.stderr, "output dir:", fparam


def train_nnlm(params):
    ftrain = params['ftrain']
    fdev = params['fdev']
    ftest = params['ftest']
    fvocab = params['fvocab']
    ffreq = params['ffreq']
    ngram = params ['ngram']
    add_unk = params['add_unk']
    use_unk = params['use_unk']
    N_input_layer = params['N']
    P_projection_layer = params['P']
    H_hidden_layer = params['H']
    N_output_layer = params['N']
    learning_rate = params['learning_rate']
    L1_reg= params['L1']
    L2_reg= params['L2'] 
    n_epochs= params['n_epochs']
    batch_size= params['batch_size']
    adaptive_learning_rate = params['use_adaptive']
    fparam = params['foutparam']

    print >> sys.stderr, 'Reading Training File: ' , ftrain
    TrainData = CreateData(ftrain,fvocab,ffreq,ngram,add_unk,use_unk)
    print >> sys.stderr, 'Reading Training File: ' , fdev
    DevData = CreateData(fdev,fvocab,ffreq,ngram,False,use_unk)
    print >> sys.stderr, 'Reading Training File: ' , ftest
    TestData = CreateData(ftest,fvocab,ffreq,ngram,False,use_unk)

    #Convert data suitable for NNLM training 
    NNLMdata = load_alldata(TrainData,DevData,TestData,ngram,N_input_layer)
    
    foldmodel = params['fmodel']
    if foldmodel.strip()!="":
        OldParams = load_params_matlab(foldmodel)
    else:
        OldParams = False 
    
    print_params(foldmodel,ngram,N_input_layer,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam)
    train_mlp(NNLMdata,OldParams,ngram,N_input_layer,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam)

    return 1

if __name__ == '__main__':
    if len(sys.argv)<2:
        print >> sys.stderr, " usage : python TrainNNLM.py <config.ini> "
        sys.exit(0)

    configfile = sys.argv[1]
    params = ReadConfigFile(configfile)
    if not params:
        print >> sys.stderr, "Could not read config file... Exiting" 
        sys.exit()
    train_nnlm(params)
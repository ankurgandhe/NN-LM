import sys 
sys.dont_write_bytecode = True
from ReadConfig  import * 
from Corpus import CreateData,GetVocabAndUNK,GetPerWordPenalty, GetSynset,GetBigramPenalty , GetFreqWordPenalty 
from NNLMio import *
from mlp_ngram import train_mlp
from mlp_multilingual_ngram import train_multi_mlp


USE_SYN=False 
WordLists=[]
def print_params(foldparam,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,number_hidden_layer):
    print >> sys.stderr, "Ngram order : ", ngram
    print >> sys.stderr, "Vocab size:", N_input_layer
    print >> sys.stderr, "Projection layer:", P_projection_layer
    print >> sys.stderr, "Hidden layer:", H_hidden_layer
    print >> sys.stderr, "Number of hidden layers:",number_hidden_layer
    print >> sys.stderr,  "learning rate:", learning_rate 
    print >> sys.stderr, "adaptive:", adaptive_learning_rate
    print >> sys.stderr, "L1_reg:", L1_reg
    print >> sys.stderr, "L2_reg:",L2_reg
    print >> sys.stderr, "max epochs:", n_epochs
    print >> sys.stderr, "batch_size:", batch_size
    print >> sys.stderr, "old params dir:", foldparam
    print >> sys.stderr, "output dir:", fparam

def write_machine(foldparam,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,printMapFile,WordID,fvocab,number_hidden_layer):
    outfp = open(fparam+"/mach.desc",'w')
    print >> outfp, "Ngram order : ", ngram
    print >> outfp, "Vocab size:", N_input_layer
    print >> outfp, "Projection layer:", P_projection_layer
    print >> outfp, "Hidden layer:", H_hidden_layer
    print >> outfp, "Number of hidden layers:",number_hidden_layer
    print >> outfp, "learning rate:", learning_rate, 
    print >> outfp, "adaptive learning rate:", adaptive_learning_rate
    print >> outfp, "Feature layer size:", n_feats
    print >> outfp, "L1_reg:", L1_reg
    print >> outfp, "L2_reg:",L2_reg
    print >> outfp, "max epochs:", n_epochs
    print >> outfp, "batch_size:", batch_size
    print >> outfp, "old params dir:", foldparam
    print >> outfp, "output dir:", fparam
    if printMapFile:
        fwrite = open(fparam+"/vocab.nnid",'w')
        for w in sorted(WordID, key=WordID.get):
            print >> fwrite, w,WordID[w]
	print >> outfp, "Vocab map file:", fparam+"/vocab.nnid"
    else:
	print >> outfp, "Vocab map file:", fvocab 

def write_machine_hydra(foldparam,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,printMapFile,WordID,fvocab,number_hidden_layer):
    outfp = open(fparam+"/mach.desc.hydra",'w')
    print >> outfp,ngram
    print >> outfp,N_input_layer
    print >> outfp,n_feats
    print >> outfp, P_projection_layer
    print >> outfp, H_hidden_layer
    print >> outfp,number_hidden_layer
    print >> outfp, "learning rate:", learning_rate,
    print >> outfp, "adaptive learning rate:", adaptive_learning_rate
    print >> outfp, "Feature layer size:", n_feats
    print >> outfp, "L1_reg:", L1_reg
    print >> outfp, "L2_reg:",L2_reg
    print >> outfp, "max epochs:", n_epochs
    print >> outfp, "batch_size:", batch_size
    print >> outfp, "old params dir:", foldparam
    print >> outfp, "output dir:", fparam
    if printMapFile:
        fwrite = open(fparam+"/vocab.nnid",'w')
        for w in sorted(WordID, key=WordID.get):
            print >> fwrite, w,WordID[w]
        print >> outfp, "Vocab map file:", fparam+"/vocab.nnid"
    else:
        print >> outfp, "Vocab map file:", fvocab


def read_params(params):
    ftrain = params['ftrain']
    fdev = params['fdev']
    ftest = params['ftest']
    fvocab = params['fvocab']
    ffreq = params['ffreq']
    ftrainfeat = params['train_feature_file']
    fdevfeat = params['dev_feature_file']
    ftestfeat = params['test_feature_file']

    n_feats =params['n_features']
    ngram = params ['ngram']
    add_unk = params['add_unk']
    use_unk = params['use_unk']
    N_input_layer = params['N']
    P_projection_layer = params['P']
    H_hidden_layer = params['H']
    number_hidden_layer = params['nH']
    N_output_layer = params['N']
    learning_rate = params['learning_rate']
    L1_reg= params['L1']
    L2_reg= params['L2'] 
    n_epochs= params['n_epochs']
    batch_size= params['batch_size']
    adaptive_learning_rate = params['use_adaptive']
    fparam = params['foutparam']
    write_janus = params['write_janus']	
    foldmodel = params['fmodel']
    gpu_copy_size = params['copy_size']
    number_of_languages = params['number_of_languages']
    use_synonyms = params['use_synonyms']
    fpenalty_vocab = params['penalty_vocab']	
    fpenalty_bigram = params['penalty_bigram']
    thresh = params['penalty_thresh']
    train_projection = params['train_projection']
    new_weights = params['new_weights']
    print >> sys.stderr, "Reading Vocab files", fvocab
    WordID, UNKw,printMapFile = GetVocabAndUNK(fvocab,ffreq,ngram,add_unk,use_unk)
    print >> sys.stderr, 'Reading Training File: ' , ftrain
    TrainData,N_input_layer,N_unk = CreateData(ftrain,WordID,UNKw,ngram,add_unk,use_unk)
    print >> sys.stderr, 'Reading Dev File: ' , fdev
    DevData,N_input_layer,N_unk = CreateData(fdev,WordID,UNKw,ngram,False,use_unk)
    print >> sys.stderr, 'Reading Test File: ' , ftest
    TestData,N_input_layer,N_unk = CreateData(ftest,WordID,UNKw,ngram,False,use_unk)
    if params['write_ngram_files']:
        WriteData(TrainData, ftrain+'.'+str(ngram)+'g')
        WriteData(DevData, fdev+'.'+str(ngram)+'g')
        WriteData(TestData, ftest+'.'+str(ngram)+'g')
        print >> sys.stderr, "ngrams file written... rerun with [ write_ngram_file = False ] for training"
	sys.exit(1)
    if ftrainfeat!="" and fdevfeat!="" and ftestfeat!="":
        print >> sys.stderr, 'Reading training, dev and test Feature Files ', ftrainfeat, fdevfeat, ftestfeat
	unkid = -1 
        NNLMFeatData = load_alldata_from_file(ftrainfeat,fdevfeat,ftestfeat,ngram,n_feats*1000,unkid)
    else:
        NNLMFeatData = []
        n_feats = 0
    #Convert data suitable for NNLM training 
    if "<UNK>" in WordID:
    	unkid = WordID["<UNK>"] 
    else:
	unkid = -1 
	print >> sys.stderr, "unk id:", unkid 
    NNLMdata = load_alldata(TrainData,DevData,TestData,ngram,N_input_layer,unkid)
    if foldmodel.strip()!="":
        OldParams = load_params_matlab_multi(foldmodel)
        #write_all_params_matlab(fparam,OldParams) #pW,hW,hB,lB,lW)
    else:
        OldParams = False 
    if not os.path.exists(fparam):
        os.makedirs(fparam)
    spl_words = UNKw
    if use_unk:
	if fpenalty_vocab=="" and fpenalty_bigram=="":
            spl_words = GetFreqWordPenalty(WordID,ffreq,thresh)
	else :
	    if fpenalty_bigram=="":
    	    	spl_words = GetPerWordPenalty(WordID,fpenalty_vocab)
	    else:
                spl_words = GetBigramPenalty(fpenalty_bigram)

    if use_synonyms:
	USE_SYN = use_synonyms
	WordLists.append(WordID)

    print >> sys.stderr,  "Writing system description"
    print_params(foldmodel,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,number_hidden_layer)
    write_machine(foldmodel,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,printMapFile,WordID,fvocab,number_hidden_layer)
    write_machine_hydra(foldmodel,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,printMapFile,WordID,fvocab,number_hidden_layer)
    return (NNLMdata,NNLMFeatData,OldParams,ngram,n_feats,N_unk,N_input_layer,P_projection_layer,H_hidden_layer,number_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,gpu_copy_size,spl_words,train_projection,new_weights,use_synonyms ) 


def train_nnlm(params_list):
    number_of_languages = 0 
    Loaded_data=[]
    for param in params_list:
    	Loaded_data.append(read_params(param))
	number_of_languages = number_of_languages + 1 

    synset =0 # {}
    if USE_SYN or WordLists!=[]:
	synset = Loaded_data[0][-1] #1783 #USE_SYN #GetSynset(WordLists)
	print >> sys.stderr, "Will regularize ",synset," word projections"
    elif number_of_languages>1:
	print >> sys.stderr, "Not using structured projectionLayer regression"
 
    #if number_of_languages==0:
    #    train_mlp(*Loaded_data[0]) #NNLMdata,NNLMFeatData,OldParams,ngram,n_feats,N_unk,N_input_layer,P_projection_layer,H_hidden_layer,number_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,gpu_copy_size)
    #else:
    NNLMdata_list = [];NNLMFeatData_list = [];OldParams_list = [];ngram_list = [];n_feats_list = [];N_unk_list = [];N_input_layer_list =  [];fparam_list = [];gpu_copy_size_list=[];spl_words_list=[];train_projection_list = [] ;new_weights_list = [] ; 
    for i in range(number_of_languages):
        NNLMdata,NNLMFeatData,OldParams,ngram,n_feats,N_unk,N_input_layer,P_projection_layer,H_hidden_layer,number_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,gpu_copy_size,spl_words,train_projection,new_weights, use_synonyms = Loaded_data[i]
        NNLMdata_list.append(NNLMdata)
        NNLMFeatData_list.append(NNLMFeatData)
        OldParams_list.append(OldParams)
        ngram_list.append(ngram)
        n_feats_list.append(n_feats)
        N_unk_list.append(N_unk)
        N_input_layer_list.append(N_input_layer)
        fparam_list.append(fparam) 
        gpu_copy_size_list.append(gpu_copy_size)	
        spl_words_list.append(spl_words)
	train_projection_list.append(train_projection)
	new_weights_list.append(new_weights)
    train_multi_mlp(NNLMdata_list,NNLMFeatData_list,OldParams_list,ngram_list,n_feats_list,N_unk_list,N_input_layer_list,P_projection_layer,H_hidden_layer,number_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam_list,gpu_copy_size_list,spl_words_list,train_projection_list,new_weights_list,synset,number_of_languages)

    '''for l in range(number_of_languages):     
    	if Loaded_data[l]['write_janus'] == True:	
	    if printMapFile:
		write_janus_LM(fparam+"/vocab.nnid",fparam,params['srilm'])
	    else:
		write_janus_LM(fvocab,fparam,params['srilm'])'''
    return 1

if __name__ == '__main__':
    if len(sys.argv)<2:
        print >> sys.stderr, " usage : python TrainNNLM.py <config.ini> "
        sys.exit(0)
    
    configfile_list = sys.argv[1:]
    param_list = []  
    for configfile in configfile_list :
	print >> sys.stderr, "Reading file", configfile 
	param = {}
    	param = ReadConfigFile(configfile)
    	if not param:
            print >> sys.stderr, "Could not read config file... Exiting" 
            sys.exit()
	param_list.append(param.copy())

    train_nnlm(param_list)

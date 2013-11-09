import sys 
sys.dont_write_bytecode = True
from ConfigParser import SafeConfigParser
import os
RequiredValues = {'inputs':['train_file','dev_file','test_file','vocab_file'], #,'vocab_freq_file'],
                  'outputs':['output_model_dir'],
                  'training_params': ['ngram','projection_layer_size']}

params = {}
def ReadConfigFile(fConfig):
    
    parser = SafeConfigParser()
    parser.read(fConfig)
    for section in [ 'inputs', 'outputs','training_params' ]:
         if not (parser.has_section(section)):
             print '%s section does not exists' % (section) 
	     print fConfig
             return False
         for value in RequiredValues[section]:
             if not parser.has_option(section, value):
                 print '%s section does not contain %s value' % (section,value)
                 return False
    
    value = parser.get('inputs', 'train_file').strip('"')
    params['ftrain'] = value
    value = parser.get('inputs', 'dev_file').strip('"')
    fdev = value 
    params['fdev']=value
    value = parser.get('inputs', 'test_file').strip('"')
    ftest = value 
    params['ftest']=value
    value = parser.get('inputs', 'vocab_file').strip('"')
    fvocab = value 
    params['fvocab']=value
    if  parser.has_option('inputs', 'vocab_freq_file'):
	value = parser.get('inputs', 'vocab_freq_file').strip('"')
    else:
	value = "" 
    ffreq = value
    params ['ffreq']   = value
    if parser.has_option('inputs', 'old_model_dir'):
        value = parser.get('inputs', 'old_model_dir').strip('"')
        fmodel = value
        params['fmodel']=value
    else:
        fmodel = ""
        params['fmodel']=""

    if parser.has_option('inputs', 'train_feature_file'):
        value = parser.get('inputs', 'train_feature_file').strip('"')
        params['train_feature_file']=value
        params['n_features'] =  parser.getint('training_params', 'n_features')
    else:
        params['train_feature_file']=""
        params['n_features'] =0

    if parser.has_option('inputs', 'dev_feature_file'):
        value = parser.get('inputs', 'dev_feature_file').strip('"')
        params['dev_feature_file']=value
    else:
        params['dev_feature_file']=""

    if parser.has_option('inputs', 'test_feature_file'):
        value = parser.get('inputs', 'test_feature_file').strip('"')
        params['test_feature_file']=value

    else:
        params['test_feature_file']=""

    value = parser.get('outputs', 'output_model_dir').strip('"')
    foutparam = value
    params['foutparam']=value
    value = parser.getint('training_params', 'ngram')
    ngram = value
    params['ngram']=value
    if parser.has_option('training_params','input_layer_size'):
    	value = parser.getint('training_params', 'input_layer_size')
    
    else:
	value = -1	
    params['N']=value
    value = parser.getint('training_params', 'projection_layer_size')
    params['P']=value
    value = parser.getint('training_params', 'hidden_layer_size')
    params['H']=value
	
    if parser.has_option('training_params','number_hidden_layer'):
	value = parser.getint('training_params','number_hidden_layer')
    else:
	value = 1
    params['nH'] = value 
	
    if parser.has_option('training_params','add_singleton_as_unk'):
        add_unk =  parser.getboolean('training_params','add_singleton_as_unk')
        params['add_unk']=add_unk
    else:
        add_unk = True # default value 
        params['add_unk']=add_unk
    if parser.has_option('training_params','use_singleton_as_unk'):    
        use_unk  =  parser.getboolean('training_params','use_singleton_as_unk')
        params['use_unk']=use_unk
    else:
        use_unk = False #default value 
        params['use_unk']=use_unk

    if parser.has_option('training_params','write_ngram_files'):
        write_files  =  parser.getboolean('training_params','write_ngram_files')
        params['write_ngram_files']=write_files
    else:
        write_files = False #default value
        params['write_ngram_files']=write_files

      
 
    if parser.has_option('training_params','use_adaptive_rate'):
        use_adaptive = parser.getboolean('training_params','use_adaptive_rate')
        params['use_adaptive']= use_adaptive
    else:
        use_adaptive = True 
        params['use_adaptive']=use_adaptive
    if parser.has_option('training_params','learning_rate'):
        rate = parser.getfloat('training_params','learning_rate')
        params['learning_rate']= rate
    else:
        learning_rate = 0.01
        params['learning_rate']=0.01
    if parser.has_option('training_params','L1_reg'):
        L1_reg = parser.getfloat('training_params','L1_reg')
        params['L1']= L1_reg
    else:
        L1 = 0.0
        params['L1']=0.0
    if parser.has_option('training_params','L2_reg'):
        L2_reg = parser.getfloat('training_params','L2_reg')
        params['L2']= L2_reg
    else:
        L2 = 0.0001
        params['L2']=0.0001
    if parser.has_option('training_params','n_epochs'):
        epochs = parser.getint('training_params','n_epochs')
        params['n_epochs']= epochs
    else:
        epochs = 500
        params['n_epochs']=500

    if parser.has_option('training_params','batch_size'):
        batch_size = parser.getint('training_params','batch_size')
        params['batch_size']= batch_size
    else:
        batch_size = 50
        params['batch_size']=50

    if parser.has_option('training_params','gpu_copy_size'):
        batch_size = parser.getint('training_params','gpu_copy_size')
        params['copy_size']= batch_size
    else:
        params['copy_size']=20000
    if parser.has_option('training_params','number_of_languages'):
        n_langs = parser.getint('training_params','number_of_languages')
        params['number_of_languages']= n_langs
    else:
        n_langs = 1
        params['number_of_languages']= n_langs

    if parser.has_option('training_params','use_synonyms'):
        use_adaptive = parser.getint('training_params','use_synonyms')
        params['use_synonyms']= use_adaptive
    else:
        use_adaptive = 0
        params['use_synonyms']=use_adaptive

    para='penalty_vocab'
    if parser.has_option('training_params',para):
        params[para] = parser.get('training_params',para).strip('"') 
    else:
        params[para]=""

    para='penalty_bigram'
    if parser.has_option('training_params',para):
        params[para] = parser.get('training_params',para).strip('"')
    else:
        params[para]=""

    para='penalty_thresh'
    if parser.has_option('training_params',para):
        params[para] = parser.getint('training_params',para)
    else:
        params[para]=-1

    para='train_projection'
    if parser.has_option('training_params',para):
        params[para] = parser.getboolean('training_params',para)
    else:
        params[para]=True

    para='new_weights'
    if parser.has_option('training_params',para):
        params[para] = parser.getboolean('training_params',para)
    else:
        params[para]=False



    if parser.has_option('outputs','write_janus'):
	write_janus = parser.getboolean('outputs','write_janus')
	fsrilm  = parser.get('outputs','srilm').strip('"')
        params['write_janus']= write_janus
	params['srilm']=fsrilm
    else:
        write_janus = False
        params['write_janus']=False

    return  params
#return (ftrain,fdev,ftest,fvocab,fmodel,foutparam,N,P,H,add_unk,use_unk,use_adaptive)


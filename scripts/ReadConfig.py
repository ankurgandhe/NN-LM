import sys 
from ConfigParser import SafeConfigParser
import os
RequiredValues = {'inputs':['train_file','dev_file','test_file','vocab_file','vocab_freq_file'],
                  'outputs':['output_model_dir'],
                  'training_params': ['ngram','input_layer_size','projection_layer_size']}

params = {}
def ReadConfigFile(fConfig):
    
    parser = SafeConfigParser()
    parser.read(fConfig)
    for section in [ 'inputs', 'outputs','training_params' ]:
         if not (parser.has_section(section)):
             print '%s section does not exists' % (section) 
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
    value = parser.get('inputs', 'vocab_freq_file').strip('"')
    ffreq = value
    params ['ffreq']   = value
    if parser.has_option('inputs', 'old_model_dir'):
        value = parser.get('inputs', 'old_model_dir').strip('"')
        fmodel = value
        params['fmodel']=value
    else:
        fmodel = ""
        params['fmodel']=""

    value = parser.get('outputs', 'output_model_dir').strip('"')
    foutparam = value
    params['foutparam']=value
    value = parser.getint('training_params', 'ngram')
    ngram = value
    params['ngram']=value
    value = parser.getint('training_params', 'input_layer_size')
    N = value
    params['N']=value
    value = parser.getint('training_params', 'projection_layer_size')
    P = value 
    params['P']=value
    value = parser.getint('training_params', 'hidden_layer_size')
    H = value
    params['H']=value 
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

    return  params
#return (ftrain,fdev,ftest,fvocab,fmodel,foutparam,N,P,H,add_unk,use_unk,use_adaptive)

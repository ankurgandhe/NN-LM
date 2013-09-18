from NNLMio import * 

fp = sys.argv[1]
fout = sys.argv[2]

pW,hW,hB,lW,lB  = load_params_matlab_multi(fp)
write_all_params_matlab(fout,(pW,hW,hB,lW,lB))

import sys
sys.dont_write_bytecode = True
from mlp_multilingual_ngram import test_mlp

if __name__ == '__main__':
    if len(sys.argv)<3:
        print >> sys.stderr, " usage : python TestNNLM.py <testfile> <feat-file/None> <param-dir> [outfile]"
        sys.exit(0)
    testfile,featfile,paramdir = sys.argv[1:4]
    if len(sys.argv) > 4:
	outfile = sys.argv[4]
    else:
	outfile = ""
    if len(sys.argv) > 5:
        write_arpa = bool(sys.argv[5])
    else:
        write_arpa = bool(0) 
    
    test_mlp(testfile,featfile,paramdir,outfile,write_arpa = write_arpa)




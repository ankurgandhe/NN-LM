import sys
sys.dont_write_bytecode = True
import math

def Getppl(fin):
    i=0
    Sum=float(0)
    Nsample = 0
    for l in open(fin):
        py=float(l.strip())
        if py == 0:
            continue
        log_prob = math.log(py,10)
        Sum+=log_prob
        Nsample = Nsample + 1

    entropy = -1 * (Sum/float(Nsample))
    print "over",Nsample,"samples,","ppl skipping OOVs", math.pow(10,entropy), #Sum, Sum/float(Nsample), math.pow(10,entropy)



def Getppl2(fin,fin2,lambda1):
    i=0
    Sum=float(0)
    Nsample = 0
    for l1,l2 in zip(open(fin),open(fin2)):
        py1=float(l1.strip())
        py2=float(l2.strip())
        if py1 == 0 or py2==0:
            continue
        py = lambda1 * py1 + (1 - lambda1) * py2 
        log_prob = math.log(py,10)
        Sum+=log_prob
        Nsample = Nsample + 1 

    entropy = -1 * (Sum/float(Nsample))
    print  "over",Nsample,"samples,",math.pow(10,entropy) #Sum, Sum/float(Nsample), math.pow(10,entropy)

if __name__ == '__main__':
    if len(sys.argv)<2:
	print "Usage: python ppl.py input-prob-file [input-prob-file2] [lambda1] "
	sys.exit(1)
    fin = sys.argv[1]
    if len(sys.argv)>2:
    	fin2 = sys.argv[2]
    	lambda1 = float(sys.argv[3])
    	Getppl2(fin,fin2,lambda1)
    else:
    	Getppl(fin)



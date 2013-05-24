import sys
sys.dont_write_bytecode = True
import math
fin = sys.argv[1]
i=0
Sum=float(0)
Nsample = 0
for l in open(fin):
    py=float(l.strip())
    log_prob = math.log(py,10)
    Sum+=log_prob
    Nsample = Nsample + 1 

entropy = -1 * (Sum/float(Nsample))
print Sum, Sum/float(Nsample), math.pow(10,entropy)

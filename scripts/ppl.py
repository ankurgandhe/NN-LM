import sys
import math
fin, Nsample = sys.argv[1:]
i=0
Nsample = int(Nsample)
Sum=float(0)

for l in open(fin):
    py=float(l.strip())
    log_prob = math.log(py,10)
    Sum+=log_prob
    
entropy = -1 * (Sum/float(Nsample))
print Sum, Sum/float(Nsample), math.pow(10,entropy)

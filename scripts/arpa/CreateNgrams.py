import sys
fvocab,fout=sys.argv[1:]
voc=[]
for l in open(fvocab):
    w=l.strip().split()[1]
    voc.append(w)
#f1g=open(fout+".1g.txt",'w')
f2g=open(fout+".2g.txt.id",'w')
#f3g=open(fout+".3g.txt",'w')

for w1 in voc:
    #print >> f1g, w1
    for w2 in voc:
        print >> f2g, w1, w2

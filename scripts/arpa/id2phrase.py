import sys 
fvocid, fin = sys.argv[1:]
voc={}
for l in open(fvocid):
    l=l.strip().split()
    voc[(l[1])]=l[0]
for l in open(fin):
    l=l.strip().split()
    for w in l:
        print voc[w],
    print ''


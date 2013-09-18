import sys
import os 
fparamdir,fout=sys.argv[1:]
voc=[]
fmach_desc = fparamdir+"/mach.desc"
desc_path = os.getcwd() 
for l in open(fmach_desc):
    l=l.strip()
    if l.find("Vocab map file")>=0:
        l=l.strip().split(':')[1]
        fvocab = desc_path + "/"+l.strip()
Word={}
for l in open(fvocab):
    l=l.strip().split()
    idx = l[1]
    w = l[0]
    Word[idx]=w
    voc.append(idx)
#f1g=open(fout+".1g.txt",'w')
f2g=open(fout+".2g.txt.id",'w')
f2gtxt = open(fout+".2g.txt",'w')
#f3g=open(fout+".3g.txt",'w')

for idx1 in voc:
    print >> f2g, idx1, "0"
    for idx2 in voc:
        print >> f2gtxt, Word[idx1], Word[idx2] 

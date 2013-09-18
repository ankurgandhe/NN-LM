'''
Create 3 grams from 2-gram pruned LM 
'''
import sys , os
f2gLM =  sys.argv[1:len(sys.argv)-1]
fparamdir = sys.argv[len(sys.argv)-1] 

voc=[]
fmach_desc = fparamdir+"/mach.desc"
desc_path = os.getcwd()
for l in open(fmach_desc):
    l=l.strip()
    if l.find("Vocab map file")>=0:
        l=l.strip().split(':')[1]
        fvocab = desc_path + "/"+l.strip()
WordID={}
for l in open(fvocab):
    l=l.strip().split()
    idx = l[1]
    w = l[0]
    WordID[w] = idx 
    voc.append(w)

start = 0 
#print >> sys.stderr, f2gLM 
for f2g in f2gLM:
  start = 0 
  for l in open(f2g):
    l=l.strip()
    if l.find("2-grams")>=0:
        start=1
        continue 
    if l.find("3-grams")>=0:
        break
    if l=="":
        continue 
    if start==1:
        l=l.split('\t')
        for w in l[1].strip().split():
            print WordID[w],
	history = l[1].strip().split()
	for w in voc:
	    print >> sys.stderr, history[0], history[1] , w 
        print 0


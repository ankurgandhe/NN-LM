import sys 
fmorph, fvocid, fvocfreq, fmorphedvoc, fNgramData = sys.argv[1:]
cid=1
classMap={}
Class={}
N=200
print >> sys.stderr,"Taking top",N,"stems as morphological suffixes/prefixes"
for l in open(fmorph):
	l=l.strip().split()
	morph=l[1].strip()
	if morph in classMap:
		continue; #Class[w]=classMap[cl]			
	else:
		classMap[morph]=cid
		cid=cid+1
		#Class[w]=classMap[cl]
	if cid >= N:
		break 
ID2Word={}
for l,m in zip(open(fvocfreq),open(fmorphedvoc)):
        l=l.strip().split()
	m=m.strip().split('+') 
	cidx=0
	cnt=0
	for w in m:
		if w.strip() in classMap:
			cidx = cidx << 8 
			cidx = cidx + classMap[w.strip()]
        w = l[1].strip()
	#ID2Word[wid]=w.strip()
	Class[w] = cidx 
for l in open(fvocid):
        l=l.strip().split()
        w = l[0].strip()	
	wid = l[1].strip()
	ID2Word[wid]=w.strip()


for l in open(fNgramData):
	l=l.strip()
	
	for wid in l.split():
		w  = ID2Word[wid]
		if w in Class:
			if Class[w]>0:
				print >> sys.stderr, ".", 
			print Class[w],
		else:
			print 0, #cid, # UNK - one extra class
	print ''
print >> sys.stderr,"Number of classes ( starts from 0):", cid 
#for w in Class:
#	print >> sys.stderr, w, Class[w]
	

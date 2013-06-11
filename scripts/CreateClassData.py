import sys 
fclass, fvocid, fNgramData = sys.argv[1:]
cid=0
classMap={}
Class={}
print >> sys.stderr,"Class file format: word<TAB>classid"
for l in open(fclass):
	l=l.strip().split('\t')
	w = l[0]
	cl = l[1]
	if cl in classMap:
		Class[w]=classMap[cl]			
	else:
		classMap[cl]=cid
		cid=cid+1
		Class[w]=classMap[cl]
ID2Word={}
for l in open(fvocid):
        l=l.strip().split()
        w = l[0]
	wid = l[1]
	ID2Word[wid]=w.strip()


for l in open(fNgramData):
	l=l.strip()
	
	for wid in l.split():
		w  = ID2Word[wid]
		if w in Class:
			print Class[w],
		else:
			print cid, # UNK - one extra class
	print ''
for w in Class:
	print >> sys.stderr, w, Class[w]
	

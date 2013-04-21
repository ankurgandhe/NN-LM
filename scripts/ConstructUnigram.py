import sys
fvoc, fSrilm = sys.argv[1:]
dict={}
start=0
for l in open(fSrilm):
	l=l.strip()	
	if l.find('2-grams')>=0:
	      break
	if l=="":
              continue 
	if l.find('1-grams')>=0:
	      start=1
	      continue 		
	if start==1:
	      l=l.strip().split()
	      dict[l[1]]=l[0]

for l in open(fvoc):
	l=l.strip().split()[0]
	if l in dict:
	      print l	
	      print dict[l]
	else:
	      print l 
	      print 0

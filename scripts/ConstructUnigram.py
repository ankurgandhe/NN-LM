import sys
sys.dont_write_bytecode = True

def ConstructUnigram(fvoc,fSrilm,fout):

	dict={}
	start=0
	fpout = open(fout,'w')
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
			print >> fpout, l	
			print >> fpout, dict[l]
		else:
			print >>fpout,  l 
			print >>fpout,  0


if __name__ == '__main__':
        fvoc, fSrilm ,fout = sys.argv[1:]
	ConstructUnigram(fvoc,fSrilm,fout)

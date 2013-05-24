import sys


fclass = sys.argv[1]
def WriteClassMap(fclass)
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

	Class['<UNK>']=cid

	return Class 

if __name__ == '__main__':
	fclass = sys.argv[1]
	Class = WriteClassMap(fclass)
	for l in Class:	
		print l
		print Class[l]
	if "<UNK>" not in Class:
		print "<UNK>"
		print cid
	

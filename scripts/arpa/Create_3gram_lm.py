import sys
import math
f1g,f1len,f2g,f2len,f3g,f3len = sys.argv[1:]
print ""
print "\\data\\" 
print "ngram 1="+str(f1len)
print "ngram 2="+str(f2len)
print "ngram 3="+str(f3len)
print ""
print "\\1-grams:"
for l in open(f1g):
    #l=l.strip().split('\t')
    #print l[0]+"\t"+l[1]+"\t"+"0"
    print l.strip() 
print ""
print "\\2-grams:"
for l in open(f2g):
    s=l.strip().split('\t') 
    if len(s)==3:
    	print l.strip() 
    else:
	print s[0]+"\t"+s[1]+"\t"+"0" 
print ""
print "\\3-grams:" 
for l in open(f3g):
    l=l.strip().split('\t')
    p10 = math.log10(float(l[0])) 
    #l[1]=l[1].replace('<UNK>','<unk>')
    print str(p10)+"\t"+l[1].strip()+"\t"+"0"

print ""
print "\\end\\" 

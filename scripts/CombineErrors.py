import sys 

for l1,l2 in zip(open(sys.argv[1]),open(sys.argv[2])):
	l1=l1.strip()
	l2=l2.strip()
	e = int(l1) + int(l2)
	if e > 0:
		print "1"
	else:
		print "0"

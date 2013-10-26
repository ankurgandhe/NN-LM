import sys 

f1,f2 = sys.argv[1:]

for p1,p2 in zip(open(f1),open(f2)):
	p1 = float(p1.strip())
        p2 = float(p2.strip())
	print p1 - p2 

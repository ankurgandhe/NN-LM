#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import argparse

def GetCharacters(fp):
	characs=[]
	for line in open(fp):
		for w in line.decode('utf-8'):
			if w not in characs:
				characs.append(w)
	return characs


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-in", "--inputfile", type=str, help="Input file")
        parser.add_argument("-out", "--outputfile", type=str, help="Output file")	
	args = parser.parse_args()
	if not args.inputfile:
		print >> sys.stderr, "Input file needs to be provided. Use -h or --help for more options"
		sys.exit()
	fp = args.inputfile
	if not args.outputfile:
		fout = sys.stdout
	else:
		fout = open(args.outputfile,'w')
	chars = GetCharacters(fp)
	for ch in chars:
		print >> fout, ch.encode('utf-8')

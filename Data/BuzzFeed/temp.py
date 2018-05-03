
from collections import Counter
import numpy as np
import pdb

f = open('BuzzFeedNewsUser.txt', 'r')

freq = {}
freq1 = {}
for line in f:
	n,u,c = map(lambda x:int(x), line.strip().split())
	if u not in freq:
		freq[u] = []
		freq[u].append(n)
	else:
		freq[u].append(n)


for k,v in freq.iteritems():
	if len(v) > 5:	
		print k
		freq1[k] = len(v)

pdb.set_trace()

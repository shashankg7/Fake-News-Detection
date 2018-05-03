import numpy as np
from scipy.io import loadmat
import os, sys
import scipy
from sklearn.model_selection import train_test_split
import pdb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as lr
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import adjusted_mutual_info_score as ami
from collections import defaultdict

buzz_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/UserFeature.mat"
buzz_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/BuzzFeedUserUser.txt"
buzz_comm = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/BuzzFeedCommunities.txt"


poli_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/UserFeature.mat"
poli_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/PolitiFactUserUser.txt"
poli_comm = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/PolitiFactCommunities.txt"



# Read feat. matrices
buzz_featvec = loadmat(buzz_feat)['X']
poli_featvec = loadmat(poli_feat)['X']
svd = NMF(n_components=100, random_state=42)
buzz_featvec = svd.fit_transform(buzz_featvec)
poli_featvec = svd.fit_transform(poli_featvec)



comm1 = defaultdict(int)
for line in open(buzz_comm, 'r'):
	u, c = map(lambda x:int(x), line.strip().split())
	comm1[c] += 1

for k, v in comm1.items():
	if v < 5:
		del comm1[k]


comm2 = defaultdict(int)
for line in open(poli_comm, 'r'):
	u, c = map(lambda x:int(x), line.strip().split())
	comm2[c] += 1


for k, v in comm2.items():
	if v < 5:
		del comm2[k]

pdb.set_trace()

n, n1 = 0, 0
for line in open(buzz_comm, 'r'):
	u, c = map(lambda x:int(x), line.strip().split())
	if c in comm1:
		n += 1

for line in open(poli_comm, 'r'):
	u, c = map(lambda x:int(x), line.strip().split())
	if c in comm2:
		n1 += 1

buzz_ground = {}
poli_ground = {}


for line in open(buzz_comm, 'r'):
	u, c = map(lambda x:int(x), line.strip().split())
	#if c in comm1:
	buzz_ground[u-1] = c


for line in open(poli_comm, 'r'):
	u, c = map(lambda x:int(x), line.strip().split())
	#if c in comm2:
	poli_ground[u-1] = c

pdb.set_trace()
buzz_featvec = buzz_featvec[buzz_ground.keys()]
poli_featvec = poli_featvec[poli_ground.keys()]

buzz_ground = buzz_ground.values()
poli_ground = poli_ground.values()

km = KMeans(n_clusters=81, n_init=1)
km1 = KMeans(n_clusters=310, n_init=1)

sc = 0.0
sc1 = 0.0
sc2 = 0.0
for i in xrange(10):
	km.fit(buzz_featvec)
	sc+=nmi(buzz_ground, km.labels_)
	sc1+=ari(buzz_ground, km.labels_)
	sc2+=ami(buzz_ground, km.labels_)


print "BUZZ"
print "nmi score %f"%(sc/float(10))
print "ari score %f"%(sc1/float(10))
print "ami score %f"%(sc2/float(10))

sc = 0.0
sc1 = 0.0
sc2 = 0.0
for i in xrange(10):
	km1.fit(poli_featvec)
	sc+=nmi(poli_ground, km1.labels_)
	sc1+=ari(poli_ground, km1.labels_)
	sc2+=ami(poli_ground, km1.labels_)


print "poli"
print "nmi score %f"%(sc/float(10))
print "ari score %f"%(sc1/float(10))
print "ami score %f"%(sc2/float(10))






pdb.set_trace()



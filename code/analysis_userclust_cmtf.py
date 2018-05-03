# Method details:
#	Comm. detection - Graph input only, std. comm. detection method 
# 	Tensor Factorization: Tried with cp_als, didn't work well
#				Tucker Decomposition giving good results


import numpy as np
from collections import defaultdict
import os, sys, pdb
from utils import parse_graphs
from sktensor import dtensor, cp_als, parafac2, tucker_hooi
from tensorly.decomposition import parafac, tucker
import tensorly as tl
import json
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
#import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import homogeneity_score, mutual_info_score, fowlkes_mallows_score

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import adjusted_mutual_info_score as ami





data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']

buzz_comm = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/BuzzFeedCommunities.txt"


poli_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/UserFeature.mat"
poli_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/PolitiFactUserUser.txt"
poli_comm = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/PolitiFactCommunities.txt"
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































def tensfact_baseline():
	n_clusters = 81
	f = open('buzz_user_tensor_45.npy')
	X_buzz = np.load(f)
	print X_buzz.shape

	X_buzz = X_buzz[buzz_ground.keys()]
	buzz_ground1 = buzz_ground.values()

	km = KMeans(n_clusters=81, init='k-means++', n_init=1, verbose=False)
	sc = 0.0
        sc1 = 0.0
        sc2 = 0.0
        for i in xrange(10):
            km.fit(X_buzz)
            sc+=nmi(buzz_ground1, km.labels_)
            sc1+=ari(buzz_ground1, km.labels_)
            sc2+=ami(buzz_ground1, km.labels_)


        print "BUZZ"
        print "nmi score %f"%(sc/float(10))
        print "ari score %f"%(sc1/float(10))
        print "ami score %f"%(sc2/float(10))












	f = open('poli_user_tensor_75.npy')
	X_poli = np.load(f)
	print X_poli.shape
	X_poli = X_poli[poli_ground.keys()]
	poli_ground1 = poli_ground.values()
	sc = 0.0
	sc1 = 0.0
	km1 = KMeans(n_clusters=310, init='k-means++', n_init=1, verbose=False)
	sc = 0.0
        sc1 = 0.0
        sc2 = 0.0
        for i in xrange(10):
            km1.fit(X_poli)
            sc+=nmi(poli_ground1, km1.labels_)
            sc1+=ari(poli_ground1, km1.labels_)
            sc2+=ami(poli_ground1, km1.labels_)


        print "poli"
        print "nmi score %f"%(sc/float(10))
        print "ari score %f"%(sc1/float(10))
        print "ami score %f"%(sc2/float(10))








if __name__ == "__main__":
	tensfact_baseline()
        #baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)





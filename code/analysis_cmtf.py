# Method details:
#	Comm. detection - Graph input only, std. comm. detection method 
# 	Tensor Factorization: Tried with cp_als, didn't work well
#				Tucker Decomposition giving good results


import numpy as np
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


data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']

def tensfact_baseline():
	n_clusters = 10
	f = open('buzz_tensor_45_10k.npy')
	T_buzz = np.load(f)
	print T_buzz.shape
	T_buzz = np.random.permutation(T_buzz)		
	km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, verbose=False)
	sc = 0.0
	for i in xrange(10):
		km.fit(T_buzz)
		#sc += metrics.silhouette_score(T_buzz, km.labels_)
		sc += metrics.calinski_harabaz_score(T_buzz, km.labels_)

	print "Silhoutte Coefficient %.3f"%(sc/float(10))


	f = open('poli_tensor_75_10k.npy')
	T_poli = np.load(f)
	print T_poli.shape
	T_poli = np.random.permutation(T_poli)
	km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, verbose=False)
	sc = 0.0
	for i in xrange(10):
		km.fit(T_poli)
		#sc += metrics.silhouette_score(T_poli, km.labels_)
		sc += metrics.calinski_harabaz_score(T_poli, km.labels_)
 
	print "Silhoutte Coefficient %.3f"%(sc/float(10))



if __name__ == "__main__":
	tensfact_baseline()
        #baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)





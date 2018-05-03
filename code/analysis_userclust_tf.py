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
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics


data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']

def tensfact_baseline():
	G_buzz, N_buzz, C_buzz, G_poli, N_poli, C_poli = parse_graphs()
	n_news1 = N_buzz.shape[0]
	n_news2 = N_poli.shape[0]
	y_buzz = [0] * n_news1
	y_poli = [0] * n_news2
	y_buzz = np.array(y_buzz)
	y_poli = np.array(y_poli)
	y_buzz[91:] = 1
	y_poli[120:] = 1
	n_clusters = 100
	if not os.path.isfile('tensor_buzz.npy'):
		T = np.zeros((N_buzz.shape[0], G_buzz.shape[0], C_buzz.shape[1]))
		n_users = G_buzz.shape[0]
		n_news = N_buzz.shape[0]
		n_comm = C_buzz.shape[1]
		for i in xrange(n_news):
			for j in xrange(n_users):
				for k in xrange(n_comm):
					T[i,j,k] = N_buzz[i,j] * C_buzz[j, k] 
		np.save('tensor_buzz.npy', T)
	else:
		f = open('tensor_buzz.npy')
		T_buzz = np.load(f)
		print T_buzz.shape
		print "Buzz tensor loaded"
		#T = dtensor(T_buzz)
		#print T.shape
		#factors = parafac(T_buzz, rank=25, init='random')
		#T_buzz = tl.tensor(T_buzz)
		# Best so far [50, 100, 5]
		core, factors = tucker(T_buzz, ranks=[45, 100, 5])
		print core.shape
		print factors[0].shape
		print factors[1].shape
		#P, fit, itr, exectimes = cp_als(T, 35, init='random')
		#P, F, D, A, fit, itr, exectimes = parafac2.parafac2(T, 10, init=42)
		# Extracting news embeddings
		#X_buzz = T_buzz
		X_buzz = factors[0]
		#X_buzz = P.U[0]
		F = open('buzz_lsi.npy', 'r')
		buzz_lsi = np.load(F)
		#X_buzz = np.hstack((X_buzz, buzz_lsi))
		print X_buzz.shape	
		#caler = MinMaxScaler()
		#X_buzz = preprocessing.scale(X_buzz)
		#X_buzz = scaler.fit_transform(X_buzz)
		#assert np.where(np.isnan(X_buzz) == True)[0].shape[0] == 0
		
		km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, verbose=False)
		print "Buzzfeed dataset's feat. extracted"
		#print X_buzz.shape 
        	X_buzz, y_buzz = shuffle(X_buzz, y_buzz, random_state=42)
		sc = 0.0
		for i in xrange(10):
			km.fit(X_buzz)
			#sc += metrics.silhouette_score(X_buzz, km.labels_)
			sc += metrics.calinski_harabaz_score(X_buzz, km.labels_)

		print "Silhoutte Coefficient %.3f"%(sc/float(10))


	if not os.path.isfile('tensor_poli.npy'):
		T = np.zeros((N_poli.shape[0], G_poli.shape[0], C_poli.shape[1]))
		n_users = G_poli.shape[0]
		n_news = N_poli.shape[0]
		n_comm = C_poli.shape[1]
		for i in xrange(n_news):
			for j in xrange(n_users):
				for k in xrange(n_comm):
					T[i,j,k] = N_poli[i,j] * C_poli[j, k] 
		np.save('tensor_poli.npy', T)
	else:
		f = open('tensor_poli.npy')
		T_poli = np.load(f)
		print T_poli.shape
		print "Politifact tensor loaded"
		T = dtensor(T_poli)
		#factors = parafac(T_poli, rank=50)
		#P, fit, itr, exectimes = cp_als(T, 35,  init='random')
		# Best so far: [50, 100, 5]
		T_poli = tl.tensor(T_poli)
		core, factors = tucker(T_poli, ranks=[45, 100, 5])
		#print " Fit value, Itr and Exectimes are:"
		#print fit
		#print itr
		#print exectimes
		# Extracting news embeddings
		X_poli = factors[0]
		#X_poli = P.U[0]
		F = open('poli_lsi.npy', 'r')
		poli_lsi = np.load(F)
		#X_poli = np.hstack((X_poli, poli_lsi))
		print X_poli.shape
		#X_buzz = preprocessing.scale(X_poli)
		#X_poli = scaler.fit_transform(X_poli)
		assert np.where(np.isnan(X_buzz) == True)[0].shape[0] == 0
		print X_poli.shape
		print "Politifact news feats. extracted"
		X_poli, y_poli = shuffle(X_poli, y_poli, random_state=42)
		km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, verbose=False)
		sc = 0.0
		for i in xrange(10):
			km.fit(X_poli)
			#sc += metrics.silhouette_score(X_poli, km.labels_)
			sc += metrics.calinski_harabaz_score(X_poli, km.labels_)
 
		print "Silhoutte Coefficient %.3f"%(sc/float(10))


	#pdb.set_trace()
	#T = dtensor(T)
	# P, fit, itr, exectimes = cp_als(T, 100,init='random')	       
        #return X_buzz, y_buzz, X_poli, y_poli


def classify(X_buzz, y_buzz, X_poli, y_poli):
 	pass       

if __name__ == "__main__":
	tensfact_baseline()
        #baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)





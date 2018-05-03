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
from sklearn.neighbors import KDTree

data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']

buzz_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/BuzzFeedNewsUser.txt"
poli_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/PolitiFactNewsUser.txt"

buzz_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/UserFeature.mat"
poli_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/UserFeature.mat"




def tensfact_baseline():
	n_clusters = 10
	f = open('buzz_user_tensor_45.npy')
	X_buzz = np.load(f)
	freq = {}
        freq_buzz = {}
	for line in open(buzz_graph, 'r'):
		n,u,c = map(lambda x:int(x)-1, line.strip().split())
		if u not in freq:
			freq[u] = []
			freq[u].append(n)
		else:
			freq[u].append(n)


	for k,v in freq.iteritems():
		#if len(v) > 5:
		freq_buzz[k] = v


	buzz_train = {}
	buzz_test = {}

	print len(freq_buzz)

	kdt = KDTree(X_buzz)
	P_1, P_5, P_10 = 0.0, 0.0, 0.0
	for k, v in freq_buzz.iteritems():
		gold_articles = freq_buzz[k]
		near_users = kdt.query(X_buzz[k].reshape(1, -1), k=15, return_distance=False)[0][1:]
		# Filter users
		near_users1 = set(near_users) & set(freq_buzz.keys())
		news_freq = {}
		#news_freq = defaultdict(int)
		for user in near_users:
			temp = freq_buzz[user]
			for t in temp:
				if t not in news_freq:
					news_freq[t] = 1
				else:
					news_freq[t] += 1
		retrieve_articles = sorted(news_freq, key=news_freq.get, reverse=True)
		prec_1 = retrieve_articles[0]
		prec_5 = retrieve_articles[:5]
		prec_10 = retrieve_articles[:10]

		p1, p5, p10 = 0.0, 0.0, 0.0
		if prec_1 in gold_articles:
			p1 += 1
		P_1 += p1
		p5 += len(set(prec_5) & set(gold_articles))/5
		P_5 += p5
		p10 += len(set(prec_10) & set(gold_articles))/10
		P_10 += p10
	
	

	print P_1/float(len(freq_buzz))
	print P_5/float(len(freq_buzz))
	print P_10/float(len(freq_buzz))


	f = open('poli_user_tensor_75.npy')
	X_poli = np.load(f)
	print X_poli.shape
	#X_poli = np.random.permutation(T_poli)
		


	

	freq = {}
	freq_poli = {}
	for line in open(poli_graph, 'r'):
		n,u,c = map(lambda x:int(x)-1, line.strip().split())
		if u not in freq:
			freq[u] = []
			freq[u].append(n)
		else:
			freq[u].append(n)


	for k,v in freq.iteritems():
		#f len(v) > 5:
		freq_poli[k] = v


	poli_train = {}
	poli_test = {}
	print len(freq_poli)

	kdt = KDTree(X_poli)
	P_1, P_5, P_10 = 0.0, 0.0, 0.0
	for k, v in freq_poli.iteritems():
		gold_articles = freq_poli[k]
		near_users = kdt.query(X_poli[k].reshape(1, -1), k=15, return_distance=False)[0][1:]
		# Filter users
		near_users1 = set(near_users) & set(freq_poli.keys())
		news_freq = {}
		#news_freq = defaultdict(int)
		for user in near_users:
			temp = freq_poli[user]
			for t in temp:
				if t not in news_freq:
					news_freq[t] = 1
				else:
					news_freq[t] += 1
		retrieve_articles = sorted(news_freq, key=news_freq.get, reverse=True)
		prec_1 = retrieve_articles[0]
		prec_5 = retrieve_articles[:5]
		prec_10 = retrieve_articles[:10]

		p1, p5, p10 = 0.0, 0.0, 0.0
		if prec_1 in gold_articles:
			p1 += 1
		P_1 += p1
		p5 += len(set(prec_5) & set(gold_articles))/5
		P_5 += p5
		
		p10 += len(set(prec_10) & set(gold_articles))/5
		P_10 += p10
		
	print P_1/float(len(freq_poli))
	print P_5/float(len(freq_poli))
	print P_10/float(len(freq_poli))
	


if __name__ == "__main__":
	tensfact_baseline()
        #baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)



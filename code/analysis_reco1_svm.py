
import numpy as np
import os, json, pdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
#from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import collections

data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']

buzz_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/BuzzFeedNewsUser.txt"
poli_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/PolitiFactNewsUser.txt"

buzz_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/UserFeature.mat"
poli_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/UserFeature.mat"

buzz_featvec = loadmat(buzz_feat)['X']
poli_featvec = loadmat(poli_feat)['X']
svd = TruncatedSVD(n_components=100)
buzz_featvec = svd.fit_transform(buzz_featvec)
poli_featvec = svd.fit_transform(poli_featvec)

def parse_newsdata():
	# Parse news json files
	X_buzz, y_buzz, X_poli, y_poli = [], [], [], []
	i = 0
	for dataset in datasets:
		dataset_dir = os.path.join(data_dir, dataset)
		fakenews_dir = os.path.join(dataset_dir, 'FakeNewsContent')
		realnews_dir = os.path.join(dataset_dir, 'RealNewsContent')
		for fakenews in os.listdir(fakenews_dir):
			if fakenews.split('.')[1] != 'py':
				#with open(os.path.join(fakenews_dir, fakenews), 'r').read() as fd:
				f = open(os.path.join(fakenews_dir, fakenews), 'r').read()
				if len(f) == 0:
					dummy_text = "no title" + " No text"
					if i == 0:
						X_buzz.append(dummy_text)
						y_buzz.append(1)
					else:
						X_poli.append(dummy_text)
						y_poli.append(1)
					continue
				data = json.loads(f)
				if i == 0:
					text = data['title'] + data['text']
					text = text.replace('\n', ' ').lower()
					X_buzz.append(text)
					
					y_buzz.append(1)
				else:
					text = data['title'] + data['text']
					text = text.replace('\n', ' ').lower()
					X_poli.append(text)
					y_poli.append(1)
		for realnews in os.listdir(realnews_dir):
			if fakenews.split('.')[1] != 'py':
				#with open(os.path.join(fakenews_dir, fakenews), 'r').read() as fd:
				f = open(os.path.join(realnews_dir, realnews), 'r').read()
				if len(f) == 0:
					dummy_text = "no title" + " No text"
					if i == 0:
						X_buzz.append(dummy_text)
						y_buzz.append(0)
					else:
						X_poli.append(dummy_text)
						y_poli.append(0)
					continue
				data = json.loads(f)
				if i == 0:
					text = data['title'] + data['text']
					text = text.replace('\n', ' ').lower()
					X_buzz.append(text)
					y_buzz.append(0)
				else:
					text = data['title'] + data['text']
					text = text.replace('\n', ' ').lower()
					X_poli.append(text)
					y_poli.append(0)
		i += 1	
	y_buzz = np.array(y_buzz)
	y_poli = np.array(y_poli)
	return X_buzz, y_buzz, X_poli, y_poli
	

def baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli):
	n_clusters = 100
	count_vec = CountVectorizer(ngram_range=(1,2), max_features=1000)
	#count_vec = TfidfVectorizer(use_idf=True, smooth_idf=True)
	X_buzz_ngram = count_vec.fit_transform(X_buzz).todense()
	X_poli_ngram = count_vec.fit_transform(X_poli).todense()
	svd = TruncatedSVD(n_components=45)
	svd1 = TruncatedSVD(n_components=75)
	X_buzz_ngram = svd.fit_transform(X_buzz_ngram)
	X_poli_ngram = svd.fit_transform(X_poli_ngram)
	n_articles_buzz = X_buzz_ngram.shape[0]
	n_articles_poli = X_poli_ngram.shape[0]
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

	
	# Reco module starts
	poli_train = {}
	poli_test = {}
	
	buzz_train = {}
	buzz_test = {}
	
	for k, v in freq_buzz.iteritems():
		user_interest = freq_buzz[k]
		user_interest_train, user_interest_test = train_test_split(user_interest, test_size=0.3)
		buzz_test[k] = user_interest_test
		buzz_train[k] = user_interest_train


	for k, v in freq_poli.iteritems():
		user_interest = freq_poli[k]
		user_interest_train, user_interest_test = train_test_split(user_interest, test_size=0.3)
		poli_test[k] = user_interest_test
		poli_train[k] = user_interest_train


	kdt = KDTree(buzz_featvec)
	kdt1 = KDTree(poli_featvec)
	P_1, P_5 = 0.0, 0.0
	for k, v in freq_buzz.iteritems():
		gold_articles = freq_buzz[k]
		near_users = kdt.query(buzz_featvec[k].reshape(1, -1), k=100, return_distance=False)[0][1:]
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
		p1, p5 = 0.0, 0.0
		if prec_1 in gold_articles:
			p1 += 1
		P_1 += p1
		p5 += len(set(prec_5) & set(gold_articles))/5
		P_5 += p5
		
	print P_1/float(len(freq_buzz))
	print P_5/float(len(freq_buzz))



	for k, v in freq_poli.iteritems():
		gold_articles = freq_poli[k]
		near_users = kdt.query(poli_featvec[k].reshape(1, -1), k=100, return_distance=False)[0][1:]
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
		p1, p5 = 0.0, 0.0
		if prec_1 in gold_articles:
			p1 += 1
		P_1 += p1
		p5 += len(set(prec_5) & set(gold_articles))/5
		P_5 += p5
		
	print P_1/float(len(freq_poli))
	print P_5/float(len(freq_poli))

		

if __name__ == "__main__":
	X_buzz, y_buzz, X_poli, y_poli = parse_newsdata()
	baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)

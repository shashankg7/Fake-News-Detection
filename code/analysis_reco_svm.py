
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
        	if len(v) > 5:
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
        	if len(v) > 5:
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
		train_neg_samples = random.sample(list( set(range(0, n_articles_buzz )) - set(user_interest_train) ),int(0.5 * len(user_interest_train)))
		train_pos = map(lambda x:(x, 1), user_interest_train)
		train_neg = map(lambda x:(x, 0), train_neg_samples)
		train_pos.extend(train_neg)
		buzz_train[k] = train_pos


	for k, v in freq_poli.iteritems():
		user_interest = freq_poli[k]
		user_interest_train, user_interest_test = train_test_split(user_interest, test_size=0.3)
		poli_test[k] = user_interest_test
		train_neg_samples = random.sample(list( set(range(0, n_articles_buzz)) - set(user_interest_train) ), int(0.5 *len(user_interest_train)))
		train_pos = map(lambda x:(x, 1), user_interest_train)
		train_neg = map(lambda x:(x, 0), train_neg_samples)
		train_pos.extend(train_neg)
		poli_train[k] = train_pos


	buzz_train_X = []
	buzz_train_y = []
	for user, articles in buzz_train.iteritems():
		for article in articles:
			temp = []
			temp.extend(buzz_featvec[user])
			temp.extend(X_buzz_ngram[article[0]])
			buzz_train_X.append(temp)
			buzz_train_y.append(article[1])
		
	
	poli_train_X = []
	poli_train_y = []
	for user, articles in poli_train.iteritems():
		temp = []
		for article in articles:
			temp = []
			temp.extend(poli_featvec[user])
			temp.extend(X_poli_ngram[article[0]])
			poli_train_X.append(temp)
			poli_train_y.append(article[1])

	poli_train_X, poli_train_y = shuffle(poli_train_X, poli_train_y)
	buzz_train_X, buzz_train_y = shuffle(buzz_train_X, buzz_train_y)

	#lr = LogisticRegression()
	lr = LinearSVC()
	lr.fit(poli_train_X, poli_train_y)

	# Testing starts
	pdb.set_trace()
	test_scores = []
	for user, articles in poli_test.iteritems():
		scores = []
		for i in xrange(X_poli_ngram.shape[0]):
			temp = []
			temp.extend(poli_featvec[user])
			temp.extend(X_poli_ngram[i])
			temp = np.array(temp).reshape(1, -1)
			scores.append(lr.decision_function(temp)[0])
			#scores.append(lr.predict_proba(temp)[0][1])
		reco_test = np.argsort(scores)	
		reco_test = reco_test.reverse()
		print reco_test	
		
		

if __name__ == "__main__":
	X_buzz, y_buzz, X_poli, y_poli = parse_newsdata()
	baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)

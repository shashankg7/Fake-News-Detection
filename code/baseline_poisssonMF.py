
import numpy as np
import os, json, pdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
import sys
from stochastic_PMF.code.pmf import PoissonMF

pdb.set_trace()


data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']


def parse_newsdata():
	# Parse news json files
	X_buzz, y_buzz, X_poli, y_poli = [], [], [], []
	i = 0
	for dataset in datasets:
		dataset_dir = os.path.join(data_dir, dataset)
		fakenews_dir = os.path.join(dataset_dir, 'FakeNewsContent')
		realnews_dir = os.path.join(dataset_dir, 'RealNewsContent')
		no_realnews = 0
		no_fakenews = 0
		no_articles = 0
		doc_ind = []
		for realnews in os.listdir(realnews_dir):
			if realnews.split('.')[1] != 'py':
				#with open(os.path.join(fakenews_dir, fakenews), 'r').read() as fd:
				f = open(os.path.join(realnews_dir, realnews), 'r').read()
				doc_ind.append(int(realnews.split('-')[0].split('_')[2])-1)
				if len(f) == 0:
					dummy_text = "no title" + " No text"
					if i == 0:
						X_buzz.append(dummy_text)
						y_buzz.append(0)
					else:
						X_poli.append(dummy_text)
						y_poli.append(0)
					no_realnews += 1
					continue
				data = json.loads(f)
				if i == 0:
					X_buzz.append(data['title'] + data['text'])
					y_buzz.append(0)
				else:
					X_poli.append(data['title'] + data['text'])
					y_poli.append(0)
				no_realnews += 1
		for fakenews in os.listdir(fakenews_dir):
			if fakenews.split('.')[1] != 'py':
				#with open(os.path.join(fakenews_dir, fakenews), 'r').read() as fd:
				f = open(os.path.join(fakenews_dir, fakenews), 'r').read()
				doc_ind.append(int(fakenews.split('-')[0].split('_')[2])-1)
				if len(f) == 0:
					dummy_text = "no title" + " No text"
					if i == 0:
						X_buzz.append(dummy_text)
						y_buzz.append(1)
					else:
						X_poli.append(dummy_text)
						y_poli.append(1)
					no_fakenews += 1
					continue
				data = json.loads(f)
				if i == 0:
					X_buzz.append(data['title'] + data['text'])
					y_buzz.append(1)
				else:
					X_poli.append(data['title'] + data['text'])
					y_poli.append(1)
				no_fakenews += 1
		no_articles = no_realnews + no_fakenews
		count_vec = CountVectorizer(ngram_range=(1,2), max_features=10000)
		#count_vec = TfidfVectorizer(use_idf=True, smooth_idf=True, ngram_range=(1,2), max_features=10000)
		#svd_model = TruncatedSVD(n_components=35)
		svd_model = NMF(n_components=25)
		#svd_model = PoissonMF(n_components=25, max_iter=1000)
		svd_transformer = Pipeline([('tfidf', count_vec), ('svd', svd_model)])
		#svd_transformer = Pipeline([('tfidf', count_vec)])
		if i == 0:
			#X_buzz_lsi = np.asarray(count_vec.fit_transform(X_buzz).todense(), dtype='float32')
			X_buzz_count = np.array(count_vec.fit_transform(X_buzz).todense())
			print X_buzz_count
			X_buzz_lsi = svd_model.fit_transform(X_buzz_count)
			print X_buzz_lsi
			#X_buzz_lsi = X_buzz_lsi.todense()
			f = open('buzz_lsi.npy', 'w')
			print X_buzz_lsi.shape
			#X1 = np.zeros_like(X_buzz_lsi)
			print no_articles
			#pdb.set_trace()
			#for j in xrange(no_articles):
			#	X1[j, :] = X_buzz_lsi[doc_ind[j], :]
			#X_buzz_lsi = X1	
			np.save(f, X_buzz_lsi)
		else:
			X_poli_count = np.array(count_vec.fit_transform(X_poli).todense())	
			#X_poli_lsi = np.asarray(count_vec.fit_transform(X_poli).todense(), dtype='float32')
			#X_poli_lsi = np.array(svd_transformer.fit_transform(X_poli).todense())
			X_poli_lsi  = svd_model.fit_transform(X_poli_count)
			#X1 = np.zeros_like(X_poli_lsi)
			f = open('poli_lsi.npy', 'w')
			#for j in xrange(no_articles):
			#	X1[j, :] = X_poli_lsi[doc_ind[j], :]	
			#print X1.shape
			#print no_articles
			#X_poli_lsi = X1
			np.save(f, X_poli_lsi)

		i += 1
	#y_buzz = np.array(y_buzz)
	#y_poli = np.array(y_poli)
	return X_buzz_lsi, y_buzz, X_poli_lsi, y_poli
	

def baseline_lsi_svm(X_buzz, y_buzz, X_poli, y_poli):
	#count_vec = CountVectorizer(ngram_range=(1,2))
	#count_vec = TfidfVectorizer(use_idf=True, smooth_idf=True)
	#svd_model = TruncatedSVD(n_components=100)
	#svd_transformer = Pipeline([('tfidf', count_vec), ('svd', svd_model)])
	#X_buzz_ngram = svd_transformer.fit_transform(X_buzz)
	#X_poli_ngram = svd_transformer.fit_transform(X_poli)	
	#T_buzz = np.asarray(X_buzz)
	#T_poli = np.asarray(X_poli_ngram)
	f = open('buzz_lsi.npy', 'w')
	np.save(f, X_buzz)
	f1 = open('poli_lsi.npy', 'w')
	np.save(f1, X_poli)
	pdb.set_trace()
	X_buzz_ngram, y_buzz = shuffle(X_buzz, y_buzz, random_state=42)
	X_poli_ngram, y_poli = shuffle(X_poli, y_poli, random_state=42)
	clf = SVC()
	#clf = XGBClassifier(n_jobs=-1)
	f1_scores = cross_val_score(clf, X_buzz_ngram, y_buzz, cv=5, scoring='f1')
	prec_scores = cross_val_score(clf, X_buzz_ngram, y_buzz, cv=5, scoring='precision')
	recall_scores = cross_val_score(clf, X_buzz_ngram, y_buzz, cv=5, scoring='recall')
	acc_scores = cross_val_score(clf, X_buzz_ngram, y_buzz, cv=5, scoring='accuracy')
	print "For BuzzFeed dataset, the results are:"
	print "F1-score is %0.3f +- %0.3f"%(f1_scores.mean(), f1_scores.std() * 2)
	print "Precision score is %0.3f +- %0.3f"%(prec_scores.mean(), prec_scores.std() * 2)
	print "Recall score is %0.3f +- %0.3f"%(recall_scores.mean(), recall_scores.std() * 2)
	print "Accuracy score is %0.3f +- %0.3f"%(acc_scores.mean(), acc_scores.std() * 2)
	
	f1_scores = cross_val_score(clf, X_poli_ngram, y_poli, cv=5, scoring='f1')
	prec_scores = cross_val_score(clf, X_poli_ngram, y_poli, cv=5, scoring='precision')
	recall_scores = cross_val_score(clf, X_poli_ngram, y_poli, cv=5, scoring='recall')
	acc_scores = cross_val_score(clf, X_poli_ngram, y_poli, cv=5, scoring='accuracy')
	print "For PolitiFact dataset, the results are:"
	print "F1-score is %0.3f +- %0.3f"%(f1_scores.mean(), f1_scores.std() * 2)
	print "Precision score is %0.3f +- %0.3f"%(prec_scores.mean(), prec_scores.std() * 2)
	print "Recall score is %0.3f +- %0.3f"%(recall_scores.mean(), recall_scores.std() * 2)
	print "Accuracy score is %0.3f +- %0.3f"%(acc_scores.mean(), acc_scores.std() * 2)


	



if __name__ == "__main__":
	X_buzz, y_buzz, X_poli, y_poli = parse_newsdata()
	baseline_lsi_svm(X_buzz, y_buzz, X_poli, y_poli)
	pdb.set_trace()

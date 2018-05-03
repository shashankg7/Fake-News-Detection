
import numpy as np
import os, json, pdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier

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
					print text
					X_poli.append(text)
					y_poli.append(0)
		i += 1	
	y_buzz = np.array(y_buzz)
	y_poli = np.array(y_poli)
	return X_buzz, y_buzz, X_poli, y_poli
	

def baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli):
	count_vec = CountVectorizer(ngram_range=(1,2), max_features=1000)
	#count_vec = TfidfVectorizer(use_idf=True, smooth_idf=True)
	X_buzz_ngram = count_vec.fit_transform(X_buzz)
	X_poli_ngram = count_vec.fit_transform(X_poli)
	pdb.set_trace()	
	X_buzz_ngram, y_buzz = shuffle(X_buzz_ngram, y_buzz, random_state=42)
	X_poli_ngram, y_poli = shuffle(X_poli_ngram, y_poli, random_state=42)
	clf = SVC()
	print X_buzz_ngram.shape
	print X_poli_ngram.shape
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
	baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)
	pdb.set_trace()

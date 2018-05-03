# Method details:
#	Comm. detection - Graph input only, std. comm. detection method 


import numpy as np
import os, sys, pdb
from utils import parse_graphs
from sktensor import dtensor, cp_als, parafac2
from tensorly.decomposition import parafac


data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']


def tensfact_baseline():
	G_buzz, N_buzz, C_buzz, G_poli, N_poli, C_poli = parse_graphs()
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
		print "Buzz tensor loaded"
		#T = dtensor(T_buzz)
		factors = parafac(T_buzz, rank=50)
		# Extracting news embeddings
		X_buzz = factors[0]
		print "Buzzfeed dataset's feat. extracted"
 
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
		print "Politifact tensor loaded"
		factors = parafac(T_poli, rank=50)
		# Extracting news embeddings
		X_poli = factors[0]
		print "Politifact news feats. extracted"

	print "Tensor written to disk"
	#pdb.set_trace()
	#T = dtensor(T)
	# P, fit, itr, exectimes = cp_als(T, 100,init='random')	
	y_buz, y_poli = [], []
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
                                        print ":(("
                                        continue
                                data = json.loads(f)
                                if i == 0:
                                        #X_buzz.append(data['title'] + data['text'])
                                        y_buzz.append(1)
                                else:
                                        #X_poli.append(data['title'] + data['text'])
                                        y_poli.append(1)
                for realnews in os.listdir(realnews_dir):
                        if fakenews.split('.')[1] != 'py':
                                #with open(os.path.join(fakenews_dir, fakenews), 'r').read() as fd:
                                f = open(os.path.join(realnews_dir, realnews), 'r').read()
                                if len(f) == 0:
                                        print ":(("
                                        continue
                                data = json.loads(f)
                                if i == 0:
                                        #X_buzz.append(data['title'] + data['text'])
                                        y_buzz.append(0)
                                else:
                                        #X_poli.append(data['title'] + data['text'])
                                        y_poli.append(0)
                i += 1
        y_buzz = np.array(y_buzz)
        y_poli = np.array(y_poli)
	pdb.set_trace()
        return X_buzz, y_buzz, X_poli, y_poli


def classify(X_buzz, y_buzz, X_poli, y_poli):
        X_buzz_ngram, y_buzz = shuffle(X_buzz, y_buzz, random_state=42)
        X_poli_ngram, y_poli = shuffle(X_poli, y_poli, random_state=42)
        clf = SVC()
        #clf = XGBClassifier(n_jobs=-1)
        f1_scores = cross_val_score(clf, X_buzz, y_buzz, cv=5, scoring='f1')
        prec_scores = cross_val_score(clf, X_buzz, y_buzz, cv=5, scoring='precision')
        recall_scores = cross_val_score(clf, X_buzz, y_buzz, cv=5, scoring='recall')
        acc_scores = cross_val_score(clf, X_buzz, y_buzz, cv=5, scoring='accuracy')
        print "For BuzzFeed dataset, the results are:"
        print "F1-score is %0.3f +- %0.3f"%(f1_scores.mean(), f1_scores.std() * 2)
        print "Precision score is %0.3f +- %0.3f"%(prec_scores.mean(), prec_scores.std() * 2)
        print "Recall score is %0.3f +- %0.3f"%(recall_scores.mean(), recall_scores.std() * 2)
        print "Accuracy score is %0.3f +- %0.3f"%(acc_scores.mean(), acc_scores.std() * 2)

        f1_scores = cross_val_score(clf, X_poli, y_poli, cv=5, scoring='f1')
        prec_scores = cross_val_score(clf, X_poli, y_poli, cv=5, scoring='precision')
        recall_scores = cross_val_score(clf, X_poli, y_poli, cv=5, scoring='recall')
        acc_scores = cross_val_score(clf, X_poli, y_poli, cv=5, scoring='accuracy')
        print "For PolitiFact dataset, the results are:"
        print "F1-score is %0.3f +- %0.3f"%(f1_scores.mean(), f1_scores.std() * 2)
        print "Precision score is %0.3f +- %0.3f"%(prec_scores.mean(), prec_scores.std() * 2)
        print "Recall score is %0.3f +- %0.3f"%(recall_scores.mean(), recall_scores.std() * 2)
        print "Accuracy score is %0.3f +- %0.3f"%(acc_scores.mean(), acc_scores.std() * 2)



if __name__ == "__main__":
	X_buzz, y_buzz, X_poli, y_poli = tensfact_baseline()
        baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)





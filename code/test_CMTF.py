# Method details:
#	Comm. detection - Graph input only, std. comm. detection method 
# 	Tensor Factorization: Tried with cp_als, didn't work well
#				Tucker Decomposition giving good results


import numpy as np
import os, sys, pdb
from utils_cesna import parse_graphs
from sktensor import dtensor, cp_als, parafac2, tucker_hooi
from tensorly.decomposition import parafac, tucker
import tensorly as tl
import json
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn import preprocessing


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
	#f = open('buzz_embedding_poisson.npy')
	f = open('buzz_tensor_45_10k.npy')
	T_buzz = np.load(f)
	print "Buzz tensor loaded"
	X_buzz = T_buzz
 	print X_buzz.shape
        X_buzz, y_buzz = shuffle(X_buzz, y_buzz, random_state=42)
	clf = SVC()
        #clf = xgb.XGBClassifier(n_jobs=-1)
	f1_scores = cross_val_score(clf, X_buzz, y_buzz, cv=5, scoring='f1')
        prec_scores = cross_val_score(clf, X_buzz, y_buzz, cv=5, scoring='precision')
	recall_scores = cross_val_score(clf, X_buzz, y_buzz, cv=5, scoring='recall')
	acc_scores = cross_val_score(clf, X_buzz, y_buzz, cv=5, scoring='accuracy')
        print "For BuzzFeed dataset, the results are:"
	print "F1-score is %0.3f +- %0.3f"%(f1_scores.mean(), f1_scores.std() * 2)
        print "Precision score is %0.3f +- %0.3f"%(prec_scores.mean(), prec_scores.std() * 2)
	print "Recall score is %0.3f +- %0.3f"%(recall_scores.mean(), recall_scores.std() * 2)
        print "Accuracy score is %0.3f +- %0.3f"%(acc_scores.mean(), acc_scores.std() * 2)


	f = open('poli_tensor_85_10k.npy')
	#f = open('poli_embeddings.npy')
	T_poli = np.load(f)
	print T_poli.shape
	print "Politifact tensor loaded"
	X_poli = T_poli
	X_poli, y_poli = shuffle(X_poli, y_poli, random_state=42)
        clf = SVC()
	#clf = xgb.XGBClassifier(n_jobs=-1)
	f1_scores = cross_val_score(clf, X_poli, y_poli, cv=5, scoring='f1')
        prec_scores = cross_val_score(clf, X_poli, y_poli, cv=5, scoring='precision')
	recall_scores = cross_val_score(clf, X_poli, y_poli, cv=5, scoring='recall')
        acc_scores = cross_val_score(clf, X_poli, y_poli, cv=5, scoring='accuracy')
	print "For PolitiFact dataset, the results are:"
       	print "F1-score is %0.3f +- %0.3f"%(f1_scores.mean(), f1_scores.std() * 2)
       	print "Precision score is %0.3f +- %0.3f"%(prec_scores.mean(), prec_scores.std() * 2)
        print "Recall score is %0.3f +- %0.3f"%(recall_scores.mean(), recall_scores.std() * 2)
        print "Accuracy score is %0.3f +- %0.3f"%(acc_scores.mean(), acc_scores.std() * 2)


	y_pred = cross_val_predict(clf, X_poli, y_poli, cv=5)
	print "For PolitiFact dataset, the results are:"
        conf_mat = confusion_matrix(y_poli, y_pred)
	print conf_mat

	#pdb.set_trace()
	#T = dtensor(T)
	# P, fit, itr, exectimes = cp_als(T, 100,init='random')	       
        #return X_buzz, y_buzz, X_poli, y_poli


def classify(X_buzz, y_buzz, X_poli, y_poli):
 	pass       

if __name__ == "__main__":
	tensfact_baseline()
        #baseline_ngram_svm(X_buzz, y_buzz, X_poli, y_poli)





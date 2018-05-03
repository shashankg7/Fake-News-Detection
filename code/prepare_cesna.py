
import numpy as np
import os, json, pdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from collections import defaultdict
import igraph
from scipy import io
import scipy

data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']


def parse_graphs():
	# Parse News-User, User-User, User-Community graphs
	i = 0
	for dataset in datasets:
		fname_u = os.path.join(os.path.join(data_dir, dataset) , 'UserFeature.mat')
		feat_file = os.path.join(os.path.join(data_dir, dataset), dataset+'nodefeat.txt')
		user_feat = io.loadmat(fname_u)['X']
		cw = scipy.sparse.coo_matrix(user_feat)
		print cw.shape
		f = open(feat_file, 'w')
		for i, j, v in zip(cw.row, cw.col, cw.data):
			text = str(i) + '\t' + str(j) + '\n'
			f.write(text)


def community_detection():
	buzz = "../Data/BuzzFeed/BuzzFeedUserUser.txt"
	poli = "../Data/PolitiFact/PolitiFactUserUser.txt"
	G = igraph.Graph.Read_Ncol(buzz)
	communities = G.community_edge_betweenness()
	no = communities.as_clustering()
	membership = no.membership


if __name__ == "__main__":
	parse_graphs()
	#ommunity_detection()

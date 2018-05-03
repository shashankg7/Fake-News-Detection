
import numpy as np
import os, json, pdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
#from xgboost import XGBClassifier
from collections import defaultdict
import igraph

data_dir = '../Data'
datasets = ['BuzzFeed', 'PolitiFact']

def parse_graphs():
	# Parse News-User, User-User, User-Community graphs
	i = 0
	for dataset in datasets:
		fname_u = os.path.join(os.path.join(data_dir, dataset) , dataset+'UserUser.txt')
		fname_n = os.path.join(os.path.join(data_dir, dataset), dataset+'NewsUser.txt')
		fname_c = os.path.join(os.path.join(data_dir, dataset), dataset+'Communities.txt')
		f_useruser = open(fname_u, 'r')
		f_newsuser = open(fname_n, 'r')
		f_usercomm = open(fname_c, 'r')
		# Storing edge list
		e = []
		# Storing news-user interactions
		nu = []
		uc = []
		n_users = -100
		n_news = -100
		n_comm = -100
		# Prune some communities. Stratergy: If #members < 5, remove the community
		comm = defaultdict(int)
		for line in f_usercomm:
			u,c = map(lambda x:int(x), line.strip().split())
			comm[c] += 1
		for k,v in comm.items():
			if v < 5:
				del comm[k]
		comm1 = []
		comm2 = {}
		print "No of comm. %f"%len(comm)
		for j in xrange(len(comm)):
			comm1.append(j)
		j = 0
		for k,v in comm.iteritems():
			comm2[j] = v
			j += 1
		comm = comm2
		f_usercomm.seek(0)
		for line in f_useruser:
			n1, n2 = map(lambda x:int(x), line.strip().split())
			e.append((n1, n2))
			if n1 >= n_users:
				n_users = n1
			elif n2 >= n_users:
				n_users = n2
		for line in f_newsuser:
			n, u, c = map(lambda x:int(x), line.strip().split())
			nu.append((n, u, c))
			if n >= n_news:
				n_news = n
		for line in f_usercomm:
			u, c = map(lambda x:int(x), line.strip().split())
			# If c is in pruned comm add
			if c in comm.keys():
				uc.append((u, c))
			else:
				uc.append((u, -1))
					#if c >= n_comm:
					#	n_comm = c
		n_comm = len(comm)
		if i == 0:
			G_buzz = np.zeros((n_users , n_users))
			N_buzz = np.zeros((n_news, n_users))
			C_buzz = np.zeros((n_users, n_comm))
			for edge in e:
				G_buzz[edge[0] -1, edge[1] -1] = 1
			for edge in nu:
				N_buzz[edge[0] -1, edge[1]-1] = edge[2]
			for edge in uc:
				# If the user belongs to pruned community make an entry, otherwise keep it 0
				if edge[1] != -1:
					C_buzz[edge[0]-1, edge[1]] = 1
				else:
					pass
		else:
			G_poli = np.zeros((n_users, n_users))
			N_poli = np.zeros((n_news, n_users))
			C_poli = np.zeros((n_users, n_comm))
			for edge in e:
				G_poli[edge[0] -1, edge[1]-1] = 1
			for edge in nu:
				N_poli[edge[0]-1, edge[1]-1] = edge[2]
			for edge in uc:
				# If the user belongs to pruned community make an entry, otherwise keep it 0
				if edge[1] != -1:
					C_poli[edge[0]-1, edge[1]] = 1
				else:
					pass
		i += 1
		print n_users
	return G_buzz, N_buzz, C_buzz, G_poli, N_poli, C_poli


def community_detection():
	buzz = "../Data/BuzzFeed/BuzzFeedUserUser.txt"
	poli = "../Data/PolitiFact/PolitiFactUserUser.txt"
	G = igraph.Graph.Read_Ncol(buzz)
	communities = G.community_edge_betweenness()
	no = communities.as_clustering()
	membership = no.membership


if __name__ == "__main__":
	#G_buzz, N_buzz, C_buzz, G_poli, N_poli, C_poli = parse_graphs()
	community_detection()

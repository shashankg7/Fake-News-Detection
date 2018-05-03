import numpy as np
from scipy.io import loadmat
import os, sys
import scipy
from sklearn.model_selection import train_test_split
import pdb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression as lr
from sklearn.decomposition import TruncatedSVD
from tensorly.decomposition import parafac, tucker
import tensorly as tl


buzz_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/UserFeature.mat"
buzz_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/BuzzFeed/BuzzFeedUserUser.txt"

poli_feat = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/UserFeature.mat"
poli_graph = "/home/shashank/FakeNewsDetection/datasets/FakeNewsNet-master/Data/PolitiFact/PolitiFactUserUser.txt"

n_samples = 10000
per = 1.0

# Read feat. matrices
#buzz_featvec = loadmat(buzz_feat)['X']
#poli_featvec = loadmat(poli_feat)['X']

f = open('tensor_buzz.npy')
T_buzz = np.load(f)
print T_buzz.shape
print "buzz tensor loaded"

core, factors = tucker(T_buzz, ranks=[45, 100, 5])
print factors[1].shape
buzz_featvec = factors[1]

f = open('tensor_poli.npy')
T_poli = np.load(f)
print T_poli.shape
print "poli tensor loaded"

core, factors = tucker(T_poli, ranks=[45, 100, 5])
print factors[1].shape
poli_featvec = factors[1]


# Temp user-user adjacency matrix
buzz_adjmat = np.zeros((buzz_featvec.shape[0], buzz_featvec.shape[0]), dtype=np.int8)
poli_adjmat = np.zeros((poli_featvec.shape[0], poli_featvec.shape[0]), dtype=np.int8)


# Read user's social graph
buzz_network = open(buzz_graph, 'r')
poli_network = open(poli_graph, 'r')


buzz_edgelist = []
poli_edgelist = []
buzz_y = []
poli_y = []
buzz_one_indices = []
poli_one_indices = []


for line in buzz_network:
	buzz_edgelist.append(map(lambda x:int(x)-1, line.strip().split()))
	temp = map(lambda x:int(x), line.strip().split())
	buzz_adjmat[temp[0]-1, temp[1]-1] = 1
	buzz_one_indices.append((temp[0]-1, temp[1]-1))
	buzz_y.append(1)


for line in poli_network:
	poli_edgelist.append(map(lambda x:int(x)-1, line.strip().split()))
	temp = map(lambda x:int(x), line.strip().split())
	poli_adjmat[temp[0]-1, temp[1]-1] = 1
	poli_one_indices.append((temp[0]-1, temp[1]-1))
	poli_y.append(1)

# Randomly sample non-zero indices from the matrix
i, j = np.nonzero(buzz_adjmat)
indices = zip(i, j)
zero_indices = np.random.choice(len(indices), len(buzz_y))
buzz_zero_indices = [indices[x] for x in zero_indices]
buzz_X = []
buzz_X.extend(buzz_one_indices)
buzz_X.extend(buzz_zero_indices)
buzz_y.extend([0] * len(buzz_zero_indices))

final_indices = np.random.choice(len(buzz_y), n_samples)
buzz_X = [buzz_X[i] for i in final_indices]
buzz_y = [buzz_y[i] for i in final_indices]


i, j = np.nonzero(poli_adjmat)
indices = zip(i, j)
zero_indices = np.random.choice(len(indices), len(poli_y))
poli_zero_indices = [indices[x] for x in zero_indices]
poli_X = []
poli_X.extend(poli_one_indices)
poli_X.extend(poli_zero_indices)
poli_y.extend([0] * len(poli_zero_indices))
final_indices = np.random.choice(len(poli_y), n_samples)
poli_X = [poli_X[i] for i in final_indices]
poli_y = [poli_y[i] for i in final_indices]

buzz_train_X, buzz_test_X, buzz_train_y, buzz_test_y = train_test_split(buzz_X, buzz_y, test_size=0.2, shuffle=True, random_state=42)
poli_train_X, poli_test_X, poli_train_y, poli_test_y = train_test_split(poli_X, poli_y, test_size=0.2, shuffle=True, random_state=42)


train_sample = np.random.choice(len(buzz_train_X), int(per * len(buzz_train_X)))
buzz_train_X = [buzz_train_X[i] for i in train_sample]
buzz_train_y = [buzz_train_y[i] for i in train_sample]

print len(buzz_train_X)

buzz_train_feats = []
buzz_test_feats = []
left_indices = map(lambda x:x[0], buzz_train_X)
right_indices = map(lambda x:x[1], buzz_train_X)
temp1 = buzz_featvec[left_indices]
temp2 = buzz_featvec[right_indices]
#buzz_train_feats = scipy.sparse.hstack((temp1, temp2))
#buzz_train_feats = scipy.sparse.csc_matrix.multiply(temp1, temp2)
buzz_train_feats = np.multiply(temp1, temp2)
#buzz_train_feats = np.hstack((temp1, temp2))

print buzz_train_feats.shape


train_sample = np.random.choice(len(poli_train_X), int(per * len(poli_train_X)))
poli_train_X = [poli_train_X[i] for i in train_sample]
poli_train_y = [poli_train_y[i] for i in train_sample]


left_indices = map(lambda x:x[0], buzz_test_X)
right_indices = map(lambda x:x[1], buzz_test_X)
temp1 = buzz_featvec[left_indices]
temp2 = buzz_featvec[right_indices]
#buzz_test_feats = scipy.sparse.hstack((temp1, temp2))
#buzz_test_feats = scipy.sparse.csc_matrix.multiply(temp1, temp2)
buzz_test_feats = np.multiply(temp1, temp2)
#buzz_test_feats = np.hstack((temp1, temp2))


left_indices = map(lambda x:x[0], poli_train_X)
right_indices = map(lambda x:x[1], poli_train_X)
temp1 = poli_featvec[left_indices]
temp2 = poli_featvec[right_indices]
#poli_train_feats = scipy.sparse.hstack((temp1, temp2))
#poli_train_feats = scipy.sparse.csc_matrix.multiply(temp1, temp2)
poli_train_feats = np.multiply(temp1, temp2)
#poli_train_feats = np.hstack((temp1, temp2))


left_indices = map(lambda x:x[0], poli_test_X)
right_indices = map(lambda x:x[1], poli_test_X)
temp1 = poli_featvec[left_indices]
temp2 = poli_featvec[right_indices]
#poli_test_feats = scipy.sparse.hstack((temp1, temp2))
#poli_test_feats = scipy.sparse.csc_matrix.multiply(temp1, temp2)
poli_test_feats = np.multiply(temp1, temp2)
#poli_test_feats = np.hstack((temp1, temp2))


test_zeros = np.where(poli_test_y == 0)
pdb.set_trace()

print "Feature extraction done!!"


pdb.set_trace()
svc = SVC()
print buzz_train_feats.shape
svc.fit(buzz_train_feats, buzz_train_y)
y_pred = svc.predict(buzz_test_feats)
print "Acc %f"%(accuracy_score(buzz_test_y, y_pred))
print "Acc %f"%(precision_score(buzz_test_y, y_pred))

pdb.set_trace()
svc = SVC()
print poli_train_feats.shape
svc.fit(poli_train_feats, poli_train_y)
y_pred = svc.predict(poli_test_feats)
print "Acc %f"%(accuracy_score(poli_test_y, y_pred))
print "Acc %f"%(precision_score(buzz_test_y, y_pred))


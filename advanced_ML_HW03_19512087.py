# -*- coding: utf-8 -*-

# DO NOT CHANGE
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt

def wkNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement PNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    dst = pairwise_distances(Xts,Xtr,metric='euclidean')
    y_pred=[]
    for i, Xts_ds in enumerate(dst):
        knn = Xts_ds[np.argsort(dst[i])[:k]]
        knn_cls = ytr[np.argsort(dst[i])[:k]]
        votes = np.zeros(len(np.unique(ytr)))
        w_j = [(knn.max() - knn[j])/(knn.max() - knn.min()) if knn.max() != knn.min() else 1 for j in range(len(knn))]
        for l in range(len(w_j)):
            votes[int(knn_cls[l])] += w_j[l]
        y_pred.append(np.argsort(votes)[-1])
    
    return y_pred

def PNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement PNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    dst = pairwise_distances(Xts,Xtr,metric='euclidean')
    y_pred=[]
    for i in range(len(dst)):
        tmp_dict = {}
        for j in np.unique(ytr):
            tmp_cls = dst[i, np.where(ytr == j)[0]]
            tmp_dict[j] = tmp_cls[np.argsort(tmp_cls)][:k]
        for l in range(len(tmp_dict)):
            for n in range(k):
                tmp_dict[l][n] /= n+1 
            tmp_dict[l] = sum(tmp_dict[l])
        y_pred.append(min(tmp_dict, key=tmp_dict.get))
    return y_pred

def accuracy_score(yts, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == yts[i]:
            correct += 1
    accuracy = correct/len(y_pred)
    return format(accuracy, ".4f")


#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot


X1,y1=datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=13)
Xtr1,Xts1, ytr1, yts1=train_test_split(X1,y1,test_size=0.2,random_state=22)

X2,y2=datasets.make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=3, n_classes=2, n_clusters_per_class=2, flip_y=0.2,random_state=75)
Xtr2, Xts2, ytr2, yts2=train_test_split(X2,y2,test_size=0.2, random_state=78)

start_time = time.time()
wkNN_y_pred_3_1 = wkNN(Xtr1,ytr1,Xts1,3,random_state=None)
wkNN_y_pred_5_1 = wkNN(Xtr1,ytr1,Xts1,5,random_state=None)
wkNN_y_pred_7_1 = wkNN(Xtr1,ytr1,Xts1,7,random_state=None)
wkNN_y_pred_9_1 = wkNN(Xtr1,ytr1,Xts1,9,random_state=None)
wkNN_y_pred_11_1 = wkNN(Xtr1,ytr1,Xts1,11,random_state=None)
PNN_y_pred_3_1 = PNN(Xtr1,ytr1,Xts1,3,random_state=None)
PNN_y_pred_5_1 = PNN(Xtr1,ytr1,Xts1,5,random_state=None)
PNN_y_pred_7_1 = PNN(Xtr1,ytr1,Xts1,7,random_state=None)
PNN_y_pred_9_1 = PNN(Xtr1,ytr1,Xts1,9,random_state=None)
PNN_y_pred_11_1 = PNN(Xtr1,ytr1,Xts1,11,random_state=None)
dataset1_time = format(time.time() - start_time, ".4f")

start_time = time.time()
wkNN_y_pred_3_2 = wkNN(Xtr2,ytr2,Xts2,3,random_state=None)
wkNN_y_pred_5_2 = wkNN(Xtr2,ytr2,Xts2,5,random_state=None)
wkNN_y_pred_7_2 = wkNN(Xtr2,ytr2,Xts2,7,random_state=None)
wkNN_y_pred_9_2 = wkNN(Xtr2,ytr2,Xts2,9,random_state=None)
wkNN_y_pred_11_2 = wkNN(Xtr2,ytr2,Xts2,11,random_state=None)
PNN_y_pred_3_2 = PNN(Xtr2,ytr2,Xts2,3,random_state=None)
PNN_y_pred_5_2 = PNN(Xtr2,ytr2,Xts2,5,random_state=None)
PNN_y_pred_7_2 = PNN(Xtr2,ytr2,Xts2,7,random_state=None)
PNN_y_pred_9_2 = PNN(Xtr2,ytr2,Xts2,9,random_state=None)
PNN_y_pred_11_2 = PNN(Xtr2,ytr2,Xts2,11,random_state=None)
dataset2_time = format(time.time() - start_time, ".4f")



print("Elapsed time: ", dataset1_time)
print("--------------------------------")
print("{:>6}".format("k"), "{:>6}".format("wkNN"), "{:>6}".format("PNN"))
print("{:>6}".format("3"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_3_1)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_3_1)))
print("{:>6}".format("5"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_5_1)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_5_1)))
print("{:>6}".format("7"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_7_1)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_7_1)))
print("{:>6}".format("9"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_9_1)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_9_1)))
print("{:>6}".format("11"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_11_1)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_11_1)))

print("\n")

print("Elapsed time: ", dataset2_time)
print("--------------------------------")
print("{:>6}".format("k"), "{:>6}".format("wkNN"), "{:>6}".format("PNN"))
print("{:>6}".format("3"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_3_2)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_3_2)))
print("{:>6}".format("5"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_5_2)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_5_2)))
print("{:>6}".format("7"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_7_2)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_7_2)))
print("{:>6}".format("9"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_9_2)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_9_2)))
print("{:>6}".format("11"), "{:>6}".format(accuracy_score(yts1, wkNN_y_pred_11_2)), "{:>6}".format(accuracy_score(yts1, PNN_y_pred_11_2)))



# Draw the figures
# Data1: k = 7

scatt = []
for j in range(200):
    if wkNN_y_pred_7_1[j]==yts1[j] :
        scatt.append(1)
    else :
        scatt.append(0)
scatt = np.array(scatt)   
missed1 = Xts1[np.where(scatt == 0), :]

scatt=[]
for j in range(200):
    if PNN_y_pred_7_1[j] == yts1[j] :
        scatt.append(1) 
    else :
        scatt.append(0)
scatt = np.array(scatt)   
missed2 = Xts1[np.where(scatt == 0), :]

plt.figure(figsize=(10, 8))
plt.scatter(Xtr1[:, 0], Xtr1[:, 1], c=ytr1, label='Train')
plt.scatter(Xts1[:, 0], Xts1[:, 1], c=yts1, marker='x', label='Test')
plt.scatter(missed1[0][:, 0], missed1[0][:, 1], marker='s', edgecolors='r', facecolors='none', label='Missclassified by wkNN')
plt.scatter(missed2[0][:, 0], missed2[0][:, 1], marker='d', edgecolors='b', facecolors='none', label='Missclassified by PNN')
plt.legend(loc='under right')


# Draw the figures
# Data2: k = 7

scatt = []
for j in range(200):
    if wkNN_y_pred_7_2[j]==yts2[j] :
        scatt.append(1)
    else :
        scatt.append(0)
scatt = np.array(scatt)   
missed3 = Xts2[np.where(scatt == 0), :]

scatt=[]
for j in range(200):
    if PNN_y_pred_7_2[j] == yts2[j] :
        scatt.append(1) 
    else :
        scatt.append(0)
scatt = np.array(scatt)   
missed4 = Xts2[np.where(scatt == 0), :]

plt.figure(figsize=(10, 8))
plt.scatter(Xtr2[:, 0], Xtr2[:, 1], c=ytr2, label='Train')
plt.scatter(Xts2[:, 0], Xts2[:, 1], c=yts2, marker='x', label='Test')
plt.scatter(missed3[0][:, 0], missed3[0][:, 1], marker='s', edgecolors='r', facecolors='none', label='Missclassified by wkNN')
plt.scatter(missed4[0][:, 0], missed4[0][:, 1], marker='d', edgecolors='b', facecolors='none', label='Missclassified by PNN')
plt.legend(loc='under right')
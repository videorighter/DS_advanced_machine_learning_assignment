# -*- coding: utf-8 -*-
# DO NOT CHANGE
import numpy as np
from itertools import product
from sklearn.svm import OneClassSVM
from scipy.sparse.csgraph import connected_components # adjacency matrix에 서 두 점이 1이면 같은 클러스터
import pandas as pd
import matplotlib.pyplot as plt

def get_adj_mat(X,svdd,num_cut):
    # X: n*p input matrix
    # svdd: trained svdd model by sci-kit learn using X
    # num_cut: number of cutting points on line segment
    #######OUTPUT########
    # return adjacent matrix size of n*n (if two points are connected A_ij=1)
    svdd.fit(X)
    X_idx = X.index
    nonbounded = np.array(X_idx[np.isin(np.arange(len(X)), svdd.support_)==False]) # if False, inside of boundary
    unbounded = svdd.support_[svdd.dual_coef_[0]<1] # boundary support vectors
    adj_set = list(product(nonbounded, unbounded))

    A = np.zeros((len(X), len(X)))
    for i in adj_set:
        l = svdd.predict(np.linspace(np.array(X.iloc[i[0]]), np.array(X.iloc[i[1]]), num_cut+1, endpoint=False))
        if -1 not in l:
            A[i[0], i[1]] = 1

    return A
    
def cluster_label(A,bsv):
    # A: adjacent matrix size of n*n (if two points are connected A_ij=1)
    # bsv: index of bounded support vectors
    #######OUTPUT########
    # return cluster labels (if samples are bounded support vectors, label=-1)
    # cluster number starts from 0 and ends to the number of clusters-1 (0, 1, ..., C-1)
    # Hint: use scipy.sparse.csgraph.connected_components

    comp_list = list(connected_components(A)[1])
    cluster_list = []
    for n in range(len(comp_list)):
        if n in bsv:
            comp_list[n] = -1

    return comp_list

ring = pd.read_csv('https://drive.google.com/uc?export=download&id=1_ygiOJ-xEPVSIvj3OzYrXtYc0Gw_Wa3a')
# support vector와 support vector가 아닌 점 사이를 20등분
num_cut=20
svdd = OneClassSVM(gamma=1, nu=0.2)
A = get_adj_mat(ring, svdd, num_cut)

bounded = svdd.support_[svdd.dual_coef_[0]==1]
comp_list = cluster_label(A, bounded)
comp_uniq = np.unique(comp_list)

xmin, xmax = ring.iloc[:,0].min()-0.5, ring.iloc[:,0].max()+0.5
ymin, ymax = ring.iloc[:,1].min()-0.5, ring.iloc[:,1].max()+0.5
X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
Z = np.c_[X.ravel(), Y.ravel()]
Z_pred = svdd.decision_function(Z)

##########Plot1###################
# Get SVG figure (draw line between two connected points with scatter plots)
# draw decision boundary
# mark differently for nsv, bsv, and free sv
plt.figure(figsize=(10, 8))
plt.xlim((-2.9, 2.9))
plt.ylim((-2.9, 2.9))
plt.contour(X, Y, Z_pred.reshape(X.shape), levels=[0], lw=2, colors='k')
plt.scatter(np.array(ring)[svdd.support_[svdd.dual_coef_[0]<1], 0], np.array(ring)[svdd.support_[svdd.dual_coef_[0]<1], 1], marker='o', c='r')
plt.scatter(np.array(ring)[svdd.support_[svdd.dual_coef_[0]<1], 0], np.array(ring)[svdd.support_[svdd.dual_coef_[0]<1], 1], marker='.', c='w', s=60)
plt.scatter(np.array(ring)[svdd.support_[svdd.dual_coef_[0]==1], 0], np.array(ring)[svdd.support_[svdd.dual_coef_[0]==1], 1], marker='x', c='b')
plt.scatter(np.array(ring)[np.isin(np.arange(len(ring)), svdd.support_)==False, 0], np.array(ring)[np.isin(np.arange(len(ring)), svdd.support_)==False, 1], marker='o', c='k')
for i in range(len(A)):
    for j in range(len(A[i])):
        if A[i][j] == 1:
            plt.plot((np.array(ring)[i][0], np.array(ring)[j][0]), (np.array(ring)[i][1], np.array(ring)[j][1]), c='k')

##########Plot2###################
# Clsuter labeling result
# different clusters should be colored using different color
# outliers (bounded support vectors) are marked with 'x'
plt.figure(figsize=(10, 8))
plt.xlim((-2.9, 2.9))
plt.ylim((-2.9, 2.9))
plt.contour(X, Y, Z_pred.reshape(X.shape), levels=[0], lw=2, colors='k')
plt.scatter(np.array(ring)[svdd.support_[svdd.dual_coef_[0]==1], 0], np.array(ring)[svdd.support_[svdd.dual_coef_[0]==1], 1], marker='x', c='b')
plt.scatter(np.array(ring)[comp_list == comp_uniq[1], 0], np.array(ring)[comp_list == comp_uniq[1], 1], marker='o', c='indigo')
plt.scatter(np.array(ring)[comp_list == comp_uniq[2], 0], np.array(ring)[comp_list == comp_uniq[2], 1], marker='o', c='slategray')
plt.scatter(np.array(ring)[comp_list == comp_uniq[3], 0], np.array(ring)[comp_list == comp_uniq[3], 1], marker='o', c='lightgreen')
plt.scatter(np.array(ring)[comp_list == comp_uniq[4], 0], np.array(ring)[comp_list == comp_uniq[4], 1], marker='o', c='gold')
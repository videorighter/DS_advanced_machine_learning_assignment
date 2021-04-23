# -*- coding: utf-8 -*-

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston

def ftest(X,y):
    n, p= X.shape
    X_data = np.concatenate((np.ones(n).reshape(-1,1), X), axis=1)
    X_T_X = X_data.T.dot(X_data)
    beta = np.linalg.inv(X_T_X).dot(X_data.T).dot(y.reshape(-1,1))
    y_hat = X_data.dot(beta)
    SSE = round(sum((y.reshape(-1, 1) - y_hat)**2)[0], 4)
    SSR = round(sum((y_hat - np.mean(y_hat))**2)[0], 4)
    SST = SSR + SSE
    MSR = round(SSR/p, 4)
    MSE = round(SSE/(n-p-1), 4)
    f_value = round(MSR/MSE, 4)
    p_value = round(1-stats.f.cdf(f_value, p, n-p-1), 4)

    
    print("-----------------------------------------------------------")
    print('%-10s'%'Factor', '%-10s'%'SS', '%-10s'%'DF', '%-10s'%'MS', '%-10s'%'F-value', '%-10s'%'pr>F')
    print('%-10s'%"Model", '%-10s'%SSR, '%-10s'%p, '%-10s'%MSR, '%-10s'%f_value, '%-10s'%p_value) 
    print('%-10s'%"Error", '%-10s'%SSE, '%-10s'%(n-p-1), '%-10s'%MSE)
    print("-----------------------------------------------------------")
    print('%-8s'%"Total", '%-8s'%SST, '%-8s'%(p+n-p-1))
    print("-----------------------------------------------------------")

def ttest(X,y,varname=None):
    name = np.append('Con',varname)
    n, p= X.shape
    X_data = np.concatenate((np.ones(n).reshape(-1,1), X), axis=1)
    X_T_X = X_data.T.dot(X_data)
    beta = np.linalg.inv(X_T_X).dot(X_data.T).dot(y.reshape(-1, 1))
    y_hat = X_data.dot(beta)
    SSE = round(sum((y.reshape(-1, 1) - y_hat)**2)[0], 4)
    SSR = round(sum((y_hat - np.mean(y_hat))**2)[0], 4)
    SST = SSR + SSE
    MSR = round(SSR/p, 4)
    MSE = round(SSE/(n-p-1), 4)
    se_sqred = MSE*(np.linalg.inv(X_T_X))
    se_sqred = np.diag(se_sqred)
    se = np.sqrt(se_sqred)
    t_value = []
    for i in range(len(se_sqred)):
        t_value.append((beta[i]/np.sqrt(se_sqred[i])))
    p_value = ((1-stats.t.cdf(np.abs(np.array(t_value)), n-p-1))*2)
    print("---------------------------------------------------")
    print('%-10s'%'Variable', '%-10s'%'coef', '%-10s'%'se', '%-10s'%'t', '%-10s'%'Pr>|t|')
    for i in range(0,14):
        print('%-10s'%name[i], '%-10s'%round(beta[i][0], 4), '%-10s'%round(se[i], 4), '%-10s'%round(t_value[i][0], 4), '%-10s'%round(p_value[i][0], 4))
    print("---------------------------------------------------")
    return 0

## Do not change!
# load data
data = load_boston()
X = data.data
y = data.target

ftest(X,y)
ttest(X,y,varname=data.feature_names)

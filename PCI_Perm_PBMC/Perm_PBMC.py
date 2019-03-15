# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:01:48 2018

@author: Inoue.S
"""

import numpy as np
from method_sub import kl_means, MLE, PCI_ell2, ImpSamp, ImpSamp_2

_ite = 1 * 10**2
_seed = 1
np.random.seed(_seed)

_date = "6.26"
_func = "sqeuclidean"

K = 2
l = 0

""" Load """
fn_X = "11.22/processX.npy"
X = np.load(fn_X)
n, d = np.shape(X)

cnt_pci = np.zeros(d)
cnt_naive = np.zeros(d)
for t in range(_ite):
    if t%100 == 0:
        print("Now Step {}".format(t))
    
    # Preprocessing
    for j in range(d):
        X[:,j] = np.random.permutation( X[:,j] )
    
    X = np.log1p(X) # log transformation
    
    # Prepare
    n_e = 1000
    index1 = np.random.choice(range(n), n_e, replace=False)
    X1 = X[index1, :]
    xi2h = MLE( X1, n_e, d )
    Xi = xi2h * np.identity(d)
    
    index2 = list( set(range(n)) - set(index1) )
    n_i = n - n_e
    X2 = X[index2, :]
    
    # Clustering
    init = np.random.choice(range(n_i), K, replace=False)
    lvec_T,c_v = kl_means(X2, K, l, init, _func, T=10)
    cT = [ lvec_T.count(k) for k in range(K) ]
    
    # Post Clustering Inference
    nap,sep,sign_tau,interval,sig = PCI_ell2(0, 1, init, lvec_T, Xi, X2, K, T=10)
    
    fg = np.isnan( sep )
    tau = np.abs( sign_tau )
    for j in range(d):
        ivl = interval[j]
        l = len(ivl)
        if l==1 and fg[j]>0:
            sep[j] = ImpSamp(sig[j], tau[j], ivl[0,0], ivl[0,1], 10**6)
        if l==2 and fg[j]>0:
            sep[j] = ImpSamp_2(sig[j], tau[j], ivl, 10**6)
        if l>2 and fg[j]>0:
            print("Step {} gene {}".format(t,j))
    
    cnt_pci += (sep < 0.05)
    cnt_naive += (nap < 0.05)

# Save 
np.savez("{}/PermPBMC_NumRej_K{}_No{}".format(_date, K, _seed), 
         cnt_pci = cnt_pci, cnt_naive = cnt_naive)
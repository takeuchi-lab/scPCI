# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 00:32:05 2019

@author: Inoue.S
"""

import numpy as np
import time
from method_gene import MLE, PCI_ell2

"""Load"""
save_dir = "Result/FACSfat/"
fn_X = "PCI_FACSfat/process_facs_fat.npy"
X = np.load(fn_X)
n, d = np.shape(X)

""" # of clusters """
K = 5

tmp = np.load( save_dir + "c_result_K{}.npz".format(K) )
index1 = list( tmp["index1"] )
init = list( tmp["init"] )
lvec_T = list( tmp["lvec_T"] )
index2 = list( set(range(n)) - set(index1) )

# Estimate Xi
n_e = 100
X1 = X[index1, :]
xi2h = MLE( X1, n_e, d )
Xi = xi2h * np.identity(d)

# Prepare
n_i = n - n_e
X2 = X[index2, :]

# Post Clustering Inference
for a in range(K):
    for b in range(a+1,K):
        print("start inference: {} vs. {}".format(a+1,b+1))
        start = time.time()
        nap,sep,sign_tau,interval,sig = PCI_ell2(a, b, init, lvec_T, Xi, X2, K, T=10)
        np.savez( save_dir + "each/logFACSfat_K{}_a{}b{}".format(K, a+1, b+1), 
                 nap = nap, sep = sep, sign_tau = sign_tau, interval=interval, 
                 label = lvec_T, sig = sig)
        print("Time: {}".format(time.time() - start))
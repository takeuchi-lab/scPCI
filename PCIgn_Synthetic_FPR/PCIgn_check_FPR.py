# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 19:15:41 2018

@author: Inoue.S
"""

import numpy as np
from method_gn import make_data_quick, kl_means, PCI_ell2_gn
import time

_metric = "sqeuclidean"
ite = 10**4
np.random.seed(1)
move_sample = 1
# No. of Cluster, Outliers, Steps
K = 2
l = 0
T = 100

if move_sample > 0:
    d = 10
    var_list = [20, 50, 100, 200]
    fn = "FPR_fig_gn/fig_d{}moveSample_fpr_gn".format(d)
else: 
    n = 20
    Sigma = np.identity(n)
    var_list = [10, 20, 50, 100]
    fn = "FPR_fig_gn/fig_n{}moveFeature_fpr_gn".format(n)
naive_fpr = []
pci_fpr = []
for v in var_list:
    if move_sample > 0:
        n = v
        Sigma = np.identity(n)
    else: 
        d = v
    print("gn_n{}_d{}_K{}".format(n,d,K))
    nap = []
    sep = []
    start = time.time()
    for _t in range(ite):
        # make_data
        X = make_data_quick(n, d, [], [], [], 1)
        # (k,l)-means
        init = np.random.choice(range(n), K, replace=False)
        lvec_T,c_v = kl_means(X, K, l, init, _metric, T)
        # Post Clustering Inference for Global Null
        p1,p2,_,_ = PCI_ell2_gn(0, 1, init, lvec_T, Sigma, 1, X, K, T)
        nap.append(p1)
        sep.append(p2)
    
    print("Time: {}".format(time.time() - start))
    nap = np.array(nap)
    sep = np.array(sep)
    print( "Naive FPR: {}".format(sum(nap<0.05) / ite) )
    print( "PCI FPR: {}".format(sum(sep<0.05) / ite) )
    print("==========================================\n")
    naive_fpr.append( sum(nap<0.05) / ite )
    pci_fpr.append( sum(sep<0.05) / ite )
naive_fpr = np.array(naive_fpr)
pci_fpr = np.array(pci_fpr)
np.savez(fn, naive_fpr = naive_fpr, pci_fpr = pci_fpr)
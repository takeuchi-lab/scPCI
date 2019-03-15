# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:09:43 2018

@author: Inoue.S
"""

import numpy as np
from method_sub import make_data_quick, kl_means, PCI_ell2
from progressbar import ProgressBar, Percentage, Bar
import time
import matplotlib.pyplot as plt

def EM_distance_k2(li1, li2, n):
    l1 = len( li1[0] & li2[0] ) + len( li1[1] & li2[1] )
    l2 = len( li1[0] & li2[1] ) + len( li1[1] & li2[0] )
    return 1 - ( max(l1,l2) / float(n) )

ite = 10**5
_metric = "sqeuclidean"
np.random.seed(0)
# No. of Cluster, Outliers, Steps
K = 2
l = 0
T = 100
# feature & sample
n = 50
d = 50
Sigma = np.identity(n)
Xi = np.identity(d)
###########################################
### Power
###########################################
true = [set(range(int(n/2))), set(range(int(n/2),n)) ]
mu_li = np.arange(0.5, 2.5, 0.5)
for mu in mu_li:
    print("mu: {}".format(mu))
    start = time.time()
    tpr = []
    p = ProgressBar(widgets=[Percentage(), Bar()], maxval=ite).start()
    for _t in range(ite):
        p.update(_t+1)
        X = make_data_quick(n, d, [mu], [[0,int(n/2)]], [[0,d]], 1)
        init = np.random.choice(range(n), K, replace=False)
        lvec_T,c_v = kl_means(X, K, l, init, _metric, T)
        # Calculate Index
        result = [set( np.where(np.array(lvec_T)==k)[0] ) for k in range(K)]
        EMd = EM_distance_k2(true, result, n)
        # Post Clustering Inference
        _,p2,_,_,_ = PCI_ell2(0, 1, init, lvec_T, Sigma, Xi, X, K, T)
        tpr.append([EMd, p2[0]<0.05])
    print("Time: {}".format(time.time() - start))
    p.finish()
    print()
    tpr = np.array(tpr)
    st_tpr = tpr[np.argsort(tpr[:,0])]
    
    np.savez("hetero_n{}d{}mu{}_ex4".format(n,d,mu), tpr=st_tpr)

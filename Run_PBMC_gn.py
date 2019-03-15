import numpy as np
import time
from method_cluster import MLE, PCI_ell2_gn

K = 10

"""Load"""
save_dir = "Result/PBMC/"
fn_X = "PCI_PBMC/processX.npy"
X = np.load(fn_X)
X = np.log1p(X)
n, d = np.shape(X)

tmp = np.load( save_dir + "c_result_K{}.npz".format(K) )
index1 = list( tmp["index1"] )
init = list( tmp["init"] )
lvec_T = list( tmp["lvec_T"] )

# Estimate Xi
n_e = 1000
X1 = X[index1, :]
xi2h = MLE(X1,n_e,d)

# Prepare
index2 = list( set(range(n)) - set(index1) )
n_i = n - n_e
X2 = X[index2, :]

# Post Clustering Inference
for a in range(K):
    for b in range(a+1,K):
        print("start inference: {} vs. {}".format(a+1,b+1))
        start = time.time()
        nap,sep,interval,chi = PCI_ell2_gn(a,b,init,lvec_T,xi2h,X2,K,T=10)
        np.savez( save_dir + "global/logPBMCgn_K{}_a{}b{}".format(K,a+1,b+1), 
                 nap = nap, sep = sep, chi = chi, interval=interval, label = lvec_T)
        print("Time: {}".format(time.time() - start))

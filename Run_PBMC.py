import numpy as np
import time
from method_gene import MLE, PCI_ell2

""" # of clusters """
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
xi2h = MLE( X1, n_e, d )
Xi = xi2h * np.identity(d)

# Prepare
index2 = list( set(range(n)) - set(index1) )
n_i = n - n_e
X2 = X[index2, :]

# Post Clustering Inference
[a1,a2] = [0,9]
for b in range(a1+1,K):
    print("start inference: {} vs. {}".format(a1+1,b+1))
    start = time.time()
    nap,sep,sign_tau,interval,sig = PCI_ell2(a1, b, init, lvec_T, Xi, X2, K, T=10)
    np.savez( save_dir + "each/logPBMC_K{}_a{}b{}".format(K, a1+1, b+1), 
             nap = nap, sep = sep, sign_tau = sign_tau, interval=interval, 
             label = lvec_T, sig = sig)
    print("Time: {}".format(time.time() - start))

for b in range(a2+1,K):
    print("start inference: {} vs. {}".format(a2+1,b+1))
    start = time.time()
    nap,sep,sign_tau,interval,sig = PCI_ell2(a2, b, init, lvec_T, Xi, X2, K, T=10)
    np.savez( save_dir + "each/logPBMC_K{}_a{}b{}".format(K, a2+1, b+1), 
             nap = nap, sep = sep, sign_tau = sign_tau, interval=interval, 
             label = lvec_T, sig = sig)
    print("Time: {}".format(time.time() - start))

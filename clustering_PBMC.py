# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 07:19:53 2017

@author: Inoue.S
"""
import numpy as np
from method_gene import kl_means
import seaborn as sns

# No. of clusters
K = 10
l = 0
_func = "sqeuclidean"

"""file name"""
save_dir = "Result/PBMC/"
dir_name = "PCI_PBMC/"
fn_X = dir_name + "processX.npy" # normalized by UMI only
fn_gn = dir_name + "use_gene_name.npy"
gene_name = np.load(fn_gn)

"""Load"""
X = np.load(fn_X)
X = np.log1p(X) # log transformation
n, d = np.shape(X)

# Prepare
n_e = 1000
index1 = np.random.choice(range(n), n_e, replace=False)
index2 = list( set(range(n)) - set(index1) )
n_i = n - n_e
X2 = X[index2, :]

# Decide first center based on Zheng's result
#result = np.load(dir_name + "result_tsne.npy")
#result2 = result[index2, :]
#init = []
#for k in range(1,K+1):
#    init.append( np.random.choice(np.where(result2[:,2]==k)[0], 1)[0] )
#init = np.array(init)

# Decide first center randomly
init = np.random.choice(range(n_i), K, replace=False)

# Clustering
#lvec_T,c_v = kmeans(X2, K, init, _func, T=10)
lvec_T,c_v = kl_means(X2, K, l, init, _func, T=10)
cT = [ lvec_T.count(k) for k in range(K) ]

tmp = np.load( dir_name + "c_result_K{}.npz".format(K) )
index1 = list( tmp["index1"] )
init = list( tmp["init"] )
lvec_T = list( tmp["lvec_T"] )
n = len(index1) + len(lvec_T)

from output_methods import visualize
sns.set_style("white")
visualize(lvec_T, index2, 10**4, 15)

# save result
np.savez( save_dir + "c_result_K{}".format(K) , index1 = index1, init = init, lvec_T = lvec_T)

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:13:17 2019

@author: Inoue.S
"""

import numpy as np
from method_gene import kl_means
import seaborn as sns
import pandas as pd

# No. of clusters (Fat: K=5)
K = 5
l = 0
_func = "sqeuclidean"

"""file name"""
save_dir = "Result/FACSfat/"
dir_name = "PCI_FACSfat/"
fn_X = dir_name + "LogNorm_Fat.csv" # normalized preprocessed data

"""Load"""
df = pd.read_csv(filepath_or_buffer=fn_X, sep=",")
X = df.values
n, d = np.shape(X)

# Prepare
n_e = 100
index1 = np.random.choice(range(n), n_e, replace=False)
index2 = list( set(range(n)) - set(index1) )
n_i = n - n_e
X2 = X[index2, :]

# Decide first center randomly
np.random.seed(8) # 8
init = np.random.choice(range(n_i), K, replace=False)

# Clustering
lvec_T,c_v = kl_means(X2, K, l, init, _func, T=10)
cT = [ lvec_T.count(k) for k in range(K) ]

K = 5
tmp = np.load( dir_name + "c_result_K{}.npz".format(K) )
index1 = list( tmp["index1"] )
init = list( tmp["init"] )
lvec_T = list( tmp["lvec_T"] )
n = len(index1) + len(lvec_T)
index2 = list( set(range(n)) - set(index1) )

from output_methods import visualize_FACSfat
visualize_FACSfat(lvec_T, index2, 15)

# save result
np.savez( save_dir + "c_result_K{}".format(K) , index1 = index1, init = init, lvec_T = lvec_T)

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 21:28:03 2019

@author: Inoue.S
"""

import numpy as np
import seaborn as sns

# No. of clusters (Fat: K=6)
K = 5

dir_name = "../PCI_FACSfat/"

tmp = np.load(dir_name + "c_result_K{}.npz".format(K))
index1 = list( tmp["index1"] )
init = list( tmp["init"] )
lvec_T = list( tmp["lvec_T"] )
n = len(index1) + len(lvec_T)
index2 = list( set(range(n)) - set(index1) )

sns.set_style("white")
from output_methods import visualize_FACSfat
visualize_FACSfat(lvec_T, index2, 15)

K = 5
"""Load : Inference Result"""
_fn = "logFACSfatgn_K{}".format(K)
adjust = K * (K-1) / 2.
##############################################################
### Note 10/23: 1vs2 => 38% overflow in ImpSamp
##############################################################
"""plot p-value """
table1 = np.ones((K-1,K-1))*np.NaN
table2 = np.ones((K-1,K-1))*np.NaN
for a in range(K):
    for b in range(a+1,K):
        _fn1 = dir_name + "global/" + _fn + "_a{}b{}.npz".format(a+1,b+1)
        tmp = np.load(_fn1)
        nap = min( adjust * tmp["nap"], 1.0 )
        sep = min( adjust * tmp["sep"], 1.0 )
        table1[a,b-1] = nap
        table2[a,b-1] = sep
from output_methods import draw_heatmap_gn
sns.set_style("darkgrid")
draw_heatmap_gn("adjust_"+_fn+"_heat", table1.T, table2.T, range(1,K), range(2,K+1))

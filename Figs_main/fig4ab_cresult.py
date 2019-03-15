# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 07:19:53 2017

@author: Inoue.S
"""
import numpy as np
import seaborn as sns

# No. of clusters
K = 10

dir_name = "../PCI_PBMC/"

tmp = np.load(dir_name + "c_result_K{}.npz".format(K))
index1 = list( tmp["index1"] )
init = list( tmp["init"] )
lvec_T = list( tmp["lvec_T"] )
n = len(index1) + len(lvec_T)
index2 = list( set(range(n)) - set(index1) )

#from output_methods import visualize
#sns.set_style("white")
#visualize(lvec_T, index2, 10**4, 15)

"""Load : Inference Result"""
_fn = "logPBMCgn_K{}".format(K)
# cs = ["Blues", "Greens", "Pastel1", "PiYG", "Purples", "Reds", "Spectral", "Vega10", "autumn"]
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
#sns.set(font_scale=2.0)
from output_methods import draw_heatmap_gn
sns.set_style("darkgrid")
draw_heatmap_gn("adjust_"+_fn+"_heat", table1.T, table2.T, "Blues", 
              range(1,K), range(2,K+1))

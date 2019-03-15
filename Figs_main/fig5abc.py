# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:19:21 2019

@author: Inoue.S
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

dir_name = "../PCI_PBMC/"
genes = ["ACTB", "CD247", "CD40", 
         "CTSW", "NKG7", "FCRL6", "SIGLEC7", "FCER2", "VPREB3"]

fn_gn = dir_name + "use_gene_name.npy"
gene_name = np.load(fn_gn)

fn_X = dir_name + "processX.npy" # normalized by UMI
X = np.load(fn_X)
X = np.log1p(X) # log transformation

### Plot Expression each gene
norm_X = ( X - np.mean(X, axis=0) ) / np.std(X, axis=0)

sns.set_style("white")
from output_methods import visualize_expr
ind = [ list(gene_name).index(j) for j in genes ]
#for i in range(len(genes)):
#    expr = norm_X[:, ind[i]]
#    visualize_expr(expr, genes[i], 5*10**3, 15)
    
### Plot Violin each gene
"""Load : Clustering Result"""
n, d = np.shape(X)
K = 10
tmp = np.load(dir_name + "c_result_K{}.npz".format(K))
index1 = list( tmp["index1"] )
index2 = list( set(range(n)) - set(index1) )
init = list( tmp["init"] )
lvec_T = list( tmp["lvec_T"] )

n_i = n - 1000
X2 = X[index2, :]

fg = np.array(lvec_T)

#c = ["#FB9A99","saddlebrown","yellowgreen","orchid","grey","red","#1c86ee",
#         "#8B008B","#6e8b3d","#99c9fb"]
#for i in range(len(genes)):
#    _num = ind[i]
#    _name = genes[i]
#    expr_li = [X2[fg==0,_num], X2[fg==1,_num], X2[fg==2,_num], X2[fg==3,_num], X2[fg==4,_num],
#               X2[fg==5,_num], X2[fg==6,_num], X2[fg==7,_num], X2[fg==8,_num], X2[fg==9,_num]]
#    
#    plt.figure()
#    plt.title(_name, fontsize=18)
#    plt.xticks(range(1,11))
#    plt.tick_params(labelsize=18)
#    parts = plt.violinplot(expr_li, showmeans=False, showmedians=False, showextrema=False)
#    
#    cnt = 0
#    for pc in parts['bodies']:
#        data = expr_li[cnt]
#        pc.set_facecolor(c[cnt])
#        pc.set_edgecolor('black')
#        pc.set_alpha(0.8)
#        q1,med,q3 = np.percentile(data, [25, 50, 75])
#        mean = np.mean(data)
#        
#        plt.vlines(cnt+1, q1, q3, color='k', linestyle='-', lw=3)
#        plt.scatter(cnt+1, mean, marker='o', facecolor='white', edgecolor="k", s=100, zorder=3, lw=3)
#        cnt+=1
#    
#    plt.xlabel("Cluster Index", fontsize=18)
#    plt.ylabel("log(UMI+1)", fontsize=18)
#    plt.savefig(_name+"-violin.jpeg", bbox_inches="tight", dpi=1600)
#    plt.show()
    
### Plot significant for genes
_fn = "logPBMC_K{}".format(K)
adjust = ( (K * (K-1) / 2.) * 1000 )

table1 = [ np.ones((9,9))*np.NaN for i in range(len(genes)) ]
table2 = [ np.ones((9,9))*np.NaN for i in range(len(genes)) ]
for a in range(K-1):
    for b in range(a+1,K):
        _fn1 = "../PCI_PBMC/each/" + _fn + "_a{}b{}.npz".format(a+1,b+1)
        tmp = np.load(_fn1)
        sep = tmp["sep"]
        nap = tmp["nap"]
        sub1 = nap[ind]
        sub2 = sep[ind]
        for _i in range(len(genes)):
            table1[_i][a,b-1] = min(adjust*sub1[_i], 1.000)
            table2[_i][a,b-1] = min(adjust*sub2[_i], 1.000)
sns.set_style("darkgrid")
from output_methods import draw_heatmap2
for _i in range(len(genes)):
    draw_heatmap2("adjust_" + _fn + "_{}_heat".format(genes[_i]), table1[_i].T, 
                  table2[_i].T, "Greens", range(1,10), range(2,11), genes[_i])


# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 23:19:35 2018

@author: Inoue.S
"""

import numpy as np
# from method_sub import make_data_quick, kl_means, PCI_ell2
import matplotlib.pyplot as plt
import seaborn as sns

# No. of Clusters, Outliers
K = 2
l = 0

np.random.seed(3)

n = 200
d = 2
Sigma = np.identity(n)
Xi = np.identity(d)
c = np.array([[0,0]])
_mulist = []
_nlist = []
_qlist = []
X = make_data_quick(n, d, _mulist, _nlist, _qlist, 1)

_metric = "sqeuclidean"

sns.set_style("white")
fig = plt.figure(figsize=(6,6))
#plt.text(-4.3,3.5,"d", fontdict={"fontsize":25, "fontweight": "bold"})
plt.scatter(X[:,0], X[:,1], s=50, facecolor='white', edgecolor="k", linewidths=1)
plt.scatter(c[:,0], c[:,1], s=500, marker="*", c="k")
plt.xlabel("Gene 1", size=22)
plt.ylabel("Gene 2", size=22)
plt.tick_params(labelsize=18)
plt.ylim([-3.2,3.2])
plt.xlim([-3.2,3.2])
#fig.patch.set_alpha(0.)
#plt.savefig("ex_step0_2d.pdf", bbox_inches="tight")
plt.savefig("ex_step0_2d.jpeg", bbox_inches="tight", dpi=1600)
plt.show()

#####################################################

# K-means with outliers
init = np.random.choice(range(n), K, replace=False)
lvec_T,c_v = kl_means(X, K, l, init, _metric, T=100)

_c = ["red", "deepskyblue"]
fig = plt.figure(figsize=(6,6))
for k in range(K):
    _fg = ( np.array(lvec_T) == k )
    plt.scatter(X[_fg,0], X[_fg,1], s=50, facecolor='white', edgecolor=_c[k], linewidths=1)
plt.scatter(c[:,0], c[:,1], s=500, c="k", marker="*")
plt.scatter(c_v[:,0], c_v[:,1], s=100, facecolor='white', edgecolor="k", linewidth=3)
plt.xlabel("Gene 1", size=22)
plt.ylabel("Gene 2", size=22)
plt.tick_params(labelsize=18)
plt.ylim([-3.2,3.2])
plt.xlim([-3.2,3.2])
#plt.savefig("ex_step1_2d.pdf", bbox_inches="tight")
plt.savefig("ex_step1_2d.jpeg", bbox_inches="tight", dpi=1600)
plt.show()

p1,p2,sign_tau,interval,sig = PCI_ell2(0, 1, init, lvec_T, Sigma, Xi, X, K, T=100)

#####################################################
fg = np.array(lvec_T)
fig, ax = plt.subplots(ncols=2, figsize=(7,6), sharey=True)
for j in range(d):
    axis = ax[j]
    expr_li = [ X[fg==0, j], X[fg==1, j] ]
    means = [ np.mean(expr_li[k]) for k in range(K) ]
    
    #axis.text(1.4, -3, r'$\^\Delta={:.2}$'.format(abs(means[0]-means[1])), fontsize=22)
    axis.xaxis.set_tick_params(labelsize=18)
    axis.yaxis.set_tick_params(labelsize=18)
    axis.set_title("Gene {}\n naive-$p$={:.2}\n scPCI-$p$={:.2}".format(j+1,p1[j],p2[j]), fontsize=22)
    axis.set_xticks(range(1,3))
    parts = axis.violinplot(expr_li, showmeans=False, showmedians=False, showextrema=False)
    
    cnt = 0
    for pc in parts['bodies']:
        data = expr_li[cnt]
        pc.set_facecolor(_c[cnt])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
        q1,med,q3 = np.percentile(data, [25, 50, 75])
        
        axis.vlines(cnt+1, q1, q3, color='k', linestyle='-', lw=3)
        axis.scatter(cnt+1, means[cnt], marker='o', facecolor='white', edgecolor="k", s=100, zorder=3, linewidth=3)
        #axis.scatter(cnt+1, med, marker='o', color='white', s=80, zorder=3)        
        data = expr_li[cnt]
        l = len(data)
        #axis.scatter(cnt+1+np.random.uniform(-0.15,0.15,l), data, color="k", s=5)
        cnt+=1
    
    axis.set_xlabel("Cluster index", fontsize=22)

ax[0].set_ylabel("Expression", fontsize=22)
#fig.savefig("ex_step2_2d.pdf", bbox_inches="tight")
plt.savefig("ex_step2_2d.jpeg", bbox_inches="tight", dpi=1600)
fig.show()


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:53:59 2019

@author: Inoue.S
"""
import numpy as np
# from method_sub import make_data_quick, kl_means
import matplotlib.pyplot as plt
import seaborn as sns

# No. of Clusters, Outliers
K = 2
l = 0

np.random.seed(0)
n = 14
d = 15
Sigma = np.identity(n)
Xi = np.identity(d)
_mulist = [3]
_nlist = [[0,int(n/2)]]
_qlist = [[0,5]]
X = make_data_quick(n, d, _mulist, _nlist, _qlist, 1)
pindex = np.random.permutation(range(d))
permX = X[:,pindex]
xmax = np.max(X)
xmin = np.min(X)


# PCI conceptA
plt.figure()
#plt.text(-2,n+1.5,"a", fontdict={"fontsize":30, "fontweight": "bold"})
sns.heatmap(permX[np.random.permutation(range(n))], cmap="RdBu_r", linewidths=3, square=True, xticklabels="",  yticklabels="",
            cbar_kws={"ticks":[]})
plt.text(17, 13, "high", fontsize=18)
plt.text(17, 0.5, "low", fontsize=18)
plt.xlabel("$d$ genes", fontsize=20)
plt.ylabel("$n$ cells", fontsize=20)
#plt.savefig("PCI-conceptA.pdf", bbox_inches="tight")
plt.savefig("PCI-conceptA.jpeg", bbox_inches="tight", dpi=1600)
plt.show()

# PCI conceptB
plt.subplot(2,1,1)
plt.text(-4.5, 3, r"Cluster 1", fontsize=18)
sns.heatmap(permX[:int(n/2)], cmap="RdBu_r", linewidths=3, square=True, 
                  vmax=xmax, vmin=xmin, xticklabels="",  yticklabels="", cbar=False)
plt.subplot(2,1,2)
plt.text(-4.5, 3, r"Cluster 2", fontsize=18)
sns.heatmap(permX[int(n/2):], cmap="RdBu_r", linewidths=3, square=True, 
                  vmax=xmax, vmin=xmin, xticklabels="",  yticklabels="", cbar=False)
#plt.savefig("PCI-conceptB.pdf", bbox_inches="tight")
plt.savefig("PCI-conceptB.jpeg", bbox_inches="tight", dpi=1600)
plt.show()

# PCI conceptC
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
plt.text(-4.5, 3, r"Cluster 1", fontsize=18)
sns.heatmap(permX[:int(n/2)], cmap="RdBu_r", linewidths=3, square=True, 
                  vmax=xmax, vmin=xmin, xticklabels="", yticklabels="", cbar=False)
ax.set_xticklabels(["$p_1$", "$p_2$", "$p_3$", "$p_4$", "$p_5$", "$p_6$", "$p_7$", "$p_8$", "$p_9$", "$p_{10}$", "$p_{11}$", "$p_{12}$", "$p_{13}$", "$p_{14}$", "$p_{15}$"], rotation=90)
ax.xaxis.set_ticks_position('top')
ax = fig.add_subplot(2,1,2)
plt.text(-4.5, 3, r"Cluster 2", fontsize=18)
sns.heatmap(permX[int(n/2):], cmap="RdBu_r", linewidths=3, square=True, 
                  vmax=xmax, vmin=xmin, xticklabels="", yticklabels="", cbar=False)
#plt.savefig("PCI-conceptC.jpeg", bbox_inches="tight", dpi=2000)
plt.savefig("PCI-conceptC.jpeg", bbox_inches="tight", dpi=1600)
plt.show()

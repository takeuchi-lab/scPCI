# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:42:38 2019

@author: Inoue.S
"""
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt

dir_name = "../PCI_Perm_PBMC/result/"

cnt_pci = np.zeros(10**3)
cnt_naive = np.zeros(10**3)
for i in range(1,7):
    fn_X = dir_name + "PermPBMC_NumRej_K2_No{}.npz".format(i)
    tmp = np.load(fn_X)
    cnt_pci += tmp["cnt_pci"]
    cnt_naive += tmp["cnt_naive"]

plt.figure(figsize=(8,4))
st_li = np.sort(cnt_naive/1000)[::-1]
th = np.argmax(st_li<=0.05)
plt.ylim([0,1])
plt.xlim([1,1000])
plt.xticks([1,200,400,600,800,1000], size=20)
plt.yticks(size=20)
plt.xlabel("Gene Index", size=20)
plt.ylabel("False positive rate", size=20)
#plt.title("Shuffle Data Naive (K=2)", size=20)
plt.plot([1,1001], [0.05, 0.05], "k--", lw=2)
plt.text(800,0.06,r"$\alpha$=0.05", fontsize=20)
plt.plot([th,th], [0,1], "k--", lw=1)
plt.scatter(np.arange(1,1001), st_li, s=10, facecolor="white", edgecolors="g", lw=1, label="naive")
plt.legend(loc="best", fontsize=20)
#plt.savefig("shuffle_K2_Naive.pdf", bbox_inches="tight")
plt.savefig("shuffle_K2_Naive.jpeg", bbox_inches="tight", dpi=1600)
plt.show()

plt.figure(figsize=(8,4))
st_li = np.sort(cnt_pci/1000)[::-1]
th = np.argmax(st_li<=0.05)
plt.ylim([0,1])
plt.xlim([1,1000])
plt.xticks([1,200,400,600,800,1000],size=20)
plt.yticks(size=20)
plt.xlabel("Gene Index", size=20)
plt.ylabel("False positive rate", size=20)
#plt.title("Shuffle Data scPCI (K=2)", size=20)
plt.plot([1,1001], [0.05, 0.05], "k--", lw=2)
plt.text(800,0.06,r"$\alpha$=0.05", fontsize=20)
plt.plot([th,th], [0,1], "k--", lw=1)
plt.scatter(np.arange(1,1001), st_li, s=10, facecolor="white", edgecolors="b", lw=1, label="scPCI")
plt.legend(loc="best", fontsize=20)
#plt.savefig("shuffle_K2_PCI.pdf", bbox_inches="tight")
plt.savefig("shuffle_K2_PCI.jpeg", bbox_inches="tight", dpi=1600)
plt.show()
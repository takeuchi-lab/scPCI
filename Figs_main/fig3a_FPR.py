# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:12:34 2019

@author: Inoue.S
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:49:13 2018

@author: Inoue.S
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

K = 2
move_sample = 1
fs = 20

dir_name = "../PCI_Synthetic_FPR/"
sns.set_style("white")
if move_sample>0: # FPR move sample
    d = 10
    fn = "fig_d{}moveSample_fpr".format(d)
    tmp = np.load(dir_name + fn + ".npz")
    naive = tmp["naive_fpr2"]
    pci = tmp["pci_fpr2"]
    
    std_mean_naive = np.sqrt( naive * (1. - naive) ) / 10**4
    std_mean_pci = np.sqrt( pci * (1. - pci) ) / 10**4
    
    nlist = [50, 100, 200, 500]
    fs = 20
    plt.figure(figsize=(8,4))
    plt.xlabel(r"$n$", fontsize=fs)
    plt.ylabel("False positive rate", fontsize=fs)
    plt.plot(nlist, naive, label="naive", linewidth=2, color="g")
    #plt.errorbar(nlist, naive, yerr=std_mean_naive, fmt='ko', ecolor='g')
    plt.scatter(nlist, naive, s=70, facecolor="white", edgecolors="g", lw=2, zorder=3)
    plt.plot(nlist, pci, label="scPCI", linewidth=2, color="b")
    #plt.errorbar(nlist, pci, yerr=std_mean_pci, fmt='ko', ecolor='b')
    plt.scatter(nlist, pci, s=70, facecolor="white", edgecolors="b", lw=2, zorder=3)
    plt.ylim([0,1.05])
    plt.xticks(nlist, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc="best", fontsize=fs)
    #plt.savefig(fn+".pdf", bbox_inches="tight", pad_inches=0.0)
    plt.savefig(fn+".jpeg", bbox_inches="tight", dpi=1600)
    plt.show()
    
else: # FPR move feature
    n = 100
    fn = "fig_n{}moveFeature_fpr".format(n)
    tmp = np.load(dir_name + fn + ".npz")
    naive = tmp["naive_fpr2"]
    pci = tmp["pci_fpr2"]
    
    std_mean_naive = np.sqrt( naive * (1. - naive) ) / 10**4
    std_mean_pci = np.sqrt( pci * (1. - pci) ) / 10**4
    
    dlist = [10, 20, 50, 100]
    plt.figure(figsize=(8,4))
    plt.xlabel(r"$d$", fontsize=fs)
    plt.ylabel("False positive rate", fontsize=fs)
    plt.plot(dlist, naive, label="naive", linewidth=2, color="g")
    #plt.errorbar(dlist, naive, yerr=std_mean_naive, fmt='ko', ecolor='g')
    plt.scatter(dlist, naive, s=70, facecolor="white", edgecolors="g", lw=2, zorder=3)
    plt.plot(dlist, pci, label="scPCI", linewidth=2, color="b")
    #plt.errorbar(dlist, pci, yerr=std_mean_pci, fmt='ko', ecolor='b')
    plt.scatter(dlist, pci, s=70, facecolor="white", edgecolors="b", lw=2, zorder=3)
    plt.ylim([0,1.05])
    plt.xticks(dlist, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc="best", fontsize=fs)
    #plt.savefig(fn+".pdf", bbox_inches="tight", pad_inches=0.0)
    plt.savefig(fn+".jpeg", bbox_inches="tight", dpi=1600)    
    plt.show()
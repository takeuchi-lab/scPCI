# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:17:29 2019

@author: Inoue.S
"""

import numpy as np
import matplotlib.pyplot as plt

K = 2
move_sample = 1

dir_name = "../PCIgn_Synthetic_FPR/"
if move_sample>0: # FPR move sample
    d = 10
    fn = "fig_d{}moveSample_fpr_gn".format(d)
    tmp = np.load(dir_name + fn + ".npz")
    naive = tmp["naive_fpr"]
    pci = tmp["pci_fpr"]
        
    nlist = [20, 50, 100, 200]
    fs = 20
    plt.figure()
    plt.xlabel(r"$n$", fontsize=fs)
    plt.ylabel("False positive rate", fontsize=fs)
    plt.plot(nlist, naive, label="naive $\chi^2$", linewidth=2, color="g")
    plt.scatter(nlist, naive, s=70, facecolor="white", edgecolors="g", lw=2, zorder=3)
    plt.plot(nlist, pci, label="scPCI-cluster", linewidth=2, color="b")
    plt.scatter(nlist, pci, s=70, facecolor="white", edgecolors="b", lw=2, zorder=3)
    plt.ylim([0,1.05])
    plt.xticks(nlist, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc="best", fontsize=fs)
    plt.savefig(fn+".pdf", bbox_inches="tight")
    plt.show()
    
else: # FPR move feature
    n = 20
    fn = "fig_n{}moveFeature_fpr_gn".format(n)
    tmp = np.load(dir_name + fn + ".npz")
    naive = tmp["naive_fpr"]
    pci = tmp["pci_fpr"]
    
    dlist = [10, 20, 50, 100]
    plt.figure()
    plt.xlabel(r"$d$", fontsize=fs)
    plt.ylabel("False positive rate", fontsize=fs)
    plt.plot(dlist, naive, label="naive $\chi^2$", linewidth=2, color="g")
    plt.scatter(dlist, naive, s=70, facecolor="white", edgecolors="g", lw=2, zorder=3)
    plt.plot(dlist, pci, label="scPCI-cluster", linewidth=2, color="b")
    plt.scatter(dlist, pci, s=70, facecolor="white", edgecolors="b", lw=2, zorder=3)
    plt.ylim([0,1.05])
    plt.xticks(dlist, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc="best", fontsize=fs)
    plt.savefig(fn+".pdf", bbox_inches="tight")
    plt.show()
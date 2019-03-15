# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:49:13 2018

@author: Inoue.S
"""

import numpy as np
import matplotlib.pyplot as plt

K = 2
move_sample = 0

#sns.set_style("whitegrid", {'grid.linestyle': '--'})
if move_sample>0: # FPR move sample
    d = 10
    fn = "fig_d{}moveSample_fpr_gn".format(d)
    tmp = np.load(fn+".npz")
    naive = tmp["naive_fpr"]
    pci = tmp["pci_fpr"]
    
    std_mean_naive = np.sqrt( naive * (1. - naive) ) / 10**4
    std_mean_pci = np.sqrt( pci * (1. - pci) ) / 10**4
    
    nlist = [20, 50, 100, 200]
    fs = 20
    plt.figure(figsize=(8,6))
    plt.xlabel(r"$n$", fontsize=fs)
    plt.ylabel("False Positive Rate", fontsize=fs)
    plt.plot(nlist, naive, label="naive $\chi^2$", linewidth=2, color="g")
    #plt.errorbar(nlist, naive, yerr=std_mean_naive, fmt='ko', ecolor='g')
    plt.scatter(nlist, naive, s=70, facecolor="white", edgecolors="g", lw=2, zorder=3)
    plt.plot(nlist, pci, label="scPCI-cluster", linewidth=2, color="b")
    plt.scatter(nlist, pci, s=70, facecolor="white", edgecolors="b", lw=2, zorder=3)
    #plt.errorbar(nlist, pci, yerr=std_mean_pci, fmt='ko', ecolor='b')
    plt.ylim([0,1.05])
    plt.xticks(nlist, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc="best", fontsize=fs)
    plt.savefig(fn+".pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()
    
else: # FPR move feature
    n = 20
    fn = "fig_n{}moveFeature_fpr_gn".format(n)
    tmp = np.load(fn+".npz")
    naive = tmp["naive_fpr"]
    pci = tmp["pci_fpr"]
    
    std_mean_naive = np.sqrt( naive * (1. - naive) ) / 10**4
    std_mean_pci = np.sqrt( pci * (1. - pci) ) / 10**4
    
    dlist = [10, 20, 50, 100]
    plt.figure(figsize=(8,6))
    plt.xlabel(r"$d$", fontsize=fs)
    plt.ylabel("False Positive Rate", fontsize=fs)
    plt.plot(dlist, naive, label="naive $\chi^2$", linewidth=2, color="g")
    plt.scatter(dlist, naive, s=70, facecolor="white", edgecolors="g", lw=2, zorder=3)
#    plt.errorbar(dlist, naive, yerr=std_mean_naive, fmt='ko', ecolor='g')
    plt.plot(dlist, pci, label="scPCI-cluster", linewidth=2, color="b")
    plt.scatter(dlist, pci, s=70, facecolor="white", edgecolors="b", lw=2, zorder=3)
#    plt.errorbar(dlist, pci, yerr=std_mean_pci, fmt='ko', ecolor='b')
    plt.ylim([0,1.05])
    plt.xticks(dlist, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc="best", fontsize=fs)
    plt.savefig(fn+".pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:49:13 2018

@author: Inoue.S
"""

import numpy as np
import seaborn
import matplotlib.pyplot as plt

# plot Global Null
n=500
d=10
K=2
tmp = np.load("gn_n{}_d{}_K{}.npz".format(n,d,K))
#naive1 = tmp["naive_fpr1"]
naive2 = tmp["naive_fpr2"]
#pci1 = tmp["pci_fpr1"]
pci2 = tmp["pci_fpr2"]

# FPR move sample
std_mean_naive = np.sqrt( naive2 * (1. - naive2) ) / 10**4
std_mean_pci = np.sqrt( pci2 * (1. - pci2) ) / 10**4

nlist = [50, 100, 200, 500]
fs=20
plt.figure(figsize=(8,6))
plt.xlabel(r"$n$", fontsize=fs)
plt.ylabel("False Positive Rate", fontsize=fs)
plt.plot(nlist, naive2, label="Naive", linewidth=2, color="g")
plt.errorbar(nlist, naive2, yerr=std_mean_naive, fmt='ko', ecolor='g')
plt.plot(nlist, pci2, label="scPCI", linewidth=2, color="b")
plt.errorbar(nlist, pci2, yerr=std_mean_pci, fmt='ko', ecolor='b')
plt.ylim([0,1])
plt.xticks(nlist, fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(loc="Best", fontsize=fs)
plt.savefig("fig_d{}moveSample_fpr.pdf".format(d))
plt.show()

# FPR move feature
n=100
d=100
K=2
tmp = np.load("gn_n{}_d{}_K{}.npz".format(n,d,K))
#naive1 = tmp["naive_fpr1"]
naive2 = tmp["naive_fpr2"]
#pci1 = tmp["pci_fpr1"]
pci2 = tmp["pci_fpr2"]

std_mean_naive = np.sqrt( naive2 * (1. - naive2) ) / 10**4
std_mean_pci = np.sqrt( pci2 * (1. - pci2) ) / 10**4

dlist = [10, 20, 50, 100]
plt.figure(figsize=(8,6))
plt.xlabel(r"$d$", fontsize=fs)
plt.ylabel("False Positive Rate", fontsize=fs)
plt.plot(dlist, naive2, label="Naive", linewidth=2, color="g")
plt.errorbar(dlist, naive2, yerr=std_mean_naive, fmt='ko', ecolor='g')
plt.plot(dlist, pci2, label="scPCI", linewidth=2, color="b")
plt.errorbar(dlist, pci2, yerr=std_mean_pci, fmt='ko', ecolor='b')
plt.ylim([0,1])
plt.xticks(dlist, fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(loc="Best", fontsize=fs)
plt.savefig("fig_n{}moveFeature_fpr.pdf".format(n))
plt.show()

# plot Hetero
n=100
d=100
mu=2
K=2
tmp = np.load("hetero_n{}_d{}_mu{}_K{}.npz".format(n,d,mu,K))
#naive1 = tmp["naive_pow1"]
naive2 = tmp["naive_pow2"]
#pci1 = tmp["pci_pow1"]
pci2 = tmp["pci_pow2"]

# power
qlist = [1,5,10,20]
plt.figure(figsize=(8,6))
plt.xlabel("$q_d$", fontsize=fs)
plt.ylabel("True Positive Rate", fontsize=fs)
#plt.plot(qlist, naive1[:,0], label="Naive $\ell_1$", linewidth=5, ls="--", color="r")
plt.plot(qlist, naive2[:,0], label="Naive $\ell_2$", linewidth=5, ls="--", color="g")
#plt.plot(qlist, pci1[:,0], label="PCI $\ell_1$", linewidth=5, color="r")
plt.plot(qlist, pci2[:,0], label="PCI $\ell_2$", linewidth=5, color="b")
plt.ylim([0,1])
plt.xticks(qlist, fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(loc="Best", fontsize=fs)
plt.savefig("fig_mu{}_pow.pdf".format(mu))
plt.show()

# false positive rate in signal data
plt.xlabel("$q_d$", fontsize=fs)
plt.ylabel("False Positive Rate", fontsize=fs)
#plt.plot(qlist, naive1[:,d-1], label="Naive $\ell_1$", linewidth=5, ls="--", color="r")
plt.plot(qlist, naive2[:,d-1], label="Naive $\ell_2$", linewidth=5, ls="--", color="g")
#plt.plot(qlist, pci1[:,d-1], label="PCI $\ell_1$", linewidth=5, color="r")
plt.plot(qlist, pci2[:,d-1], label="PCI $\ell_2$", linewidth=5, color="b")
plt.ylim([0,1])
plt.xticks(qlist)
plt.legend(loc="Best", fontsize=fs)
plt.savefig("fig_mu{}_fpr.pdf".format(mu))
plt.show()
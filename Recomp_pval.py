# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:21:14 2018

@author: Inoue.S
"""

import numpy as np
from method_gene import ImpSamp, ImpSamp_2

_name = "PBMC" # PBMC, FACSfat
d = 10**3

""" Recomp p-values """
if _name=="PBMC":
    dir_name = "Result/PBMC/each/"
    K = 10
    _fn = "logPBMC_K{}".format(K)
if _name=="FACSfat":
    dir_name = "Result/FACSfat/each/"
    K = 5
    _fn = "logFACSfat_K{}".format(K)

for a in range(K-1):
    for b in range(a+1,K):
        print("{} vs {}".format(a+1,b+1))
        _fn1 = dir_name + _fn + "_a{}b{}.npz".format(a+1,b+1)
        tmp = np.load(_fn1, encoding="bytes")
        nap = tmp["nap"]
        sep = tmp["sep"]
        sign_tau = tmp["sign_tau"]
        interval = tmp["interval"]
        lvec_T = tmp["label"]
        
        fg = np.isnan(sep) + (sep==0)
        tau = np.abs( tmp["sign_tau"] )
        sig = tmp["sig"]
        for i in range(d):
            ivl = interval[i]
            l = len(ivl)
            if l==1 and fg[i]>0:
                print("Stats: {}".format(tau[i]))
                print("std  : {}".format(sig[i]))
                print(ivl)
                sep[i] = ImpSamp(sig[i], tau[i], ivl[0,0], ivl[0,1], 10**8)
                print("sep  : {}".format(sep[i]))
                print("$$$$$$$$\n")
            if l==2 and fg[i]>0:
                print("Stats: {}".format(tau[i]))
                print("std  : {}".format(sig[i]))
                print(ivl)
                sep[i] = ImpSamp_2(sig[i], tau[i], ivl, 10**8)
                print("sep  : {}".format(sep[i]))
                print("$$$$$$$$\n")
                                
        np.savez(_fn1, nap = nap, sep = sep, sign_tau = sign_tau, interval=interval, label = lvec_T, sig = sig)

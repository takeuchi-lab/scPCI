# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:10:48 2019

@author: Inoue.S
"""

import numpy as np
from method_cluster import ImpSamp_gn

_name = "PBMC" # PBMC, FACSfat
d = 10**3

""" Recomp p-values """
if _name=="PBMC":
    dir_name = "Result/PBMC/global/"
    K = 10
    _fn = "logPBMCgn_K{}".format(K)
if _name=="FACSfat":
    dir_name = "Result/FACSfat/global/"
    K = 5
    _fn = "logFACSfatgn_K{}".format(K)

for a in range(K-1):
    for b in range(a+1,K):
        print("{} vs {}".format(a+1,b+1))
        _fn1 = dir_name + _fn + "_a{}b{}.npz".format(a+1,b+1)
        tmp = np.load(_fn1, encoding="bytes")
        nap = tmp["nap"]
        sep = tmp["sep"]
        chi = tmp["chi"]
        interval = tmp["interval"]
        lvec_T = tmp["label"]
                
        if np.isnan(sep) or (sep==0):
            print("Stats: {}".format(chi**2))
            print(interval**2)
            sep = ImpSamp_gn(chi**2, d, interval**2, 10**8)
            print("sep  : {}".format(sep))
            print("$$$$$$$$\n")
            np.savez(_fn1, nap = nap, sep = sep, chi = chi, interval=interval, label = lvec_T)
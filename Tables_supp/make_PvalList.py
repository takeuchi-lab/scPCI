# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:57:58 2019

@author: Inoue.S
"""

import numpy as np
from xlwt import Workbook

dataname = "FACSfat" #PBMC, FACSfat
d = 10**3

if dataname == "PBMC":
    K = 10
    dir_name = "../PCI_PBMC/"
    _fn = "logPBMC_K{}".format(K)
    gene_name = np.load(dir_name + "use_gene_name.npy")

if dataname == "FACSfat":
    K = 5
    dir_name = "../PCI_FACSfat/"
    _fn = "logFACSfat_K{}".format(K)
    gene_name = np.load(dir_name + "use_gene_name_Fat.npy")


""" Make Excel """
header1 = []
naive_p = np.empty((d,0))
pci_p = np.empty((d,0))
for a in range(K-1):
    for b in range(a+1,K):
        header1.append( "{} vs {}".format(a+1,b+1))
        _fn1 = dir_name + "each/" + _fn + "_a{}b{}.npz".format(a+1,b+1)
        tmp = np.load(_fn1)
        nap = tmp["nap"]
        sep = tmp["sep"]
        naive_p = np.c_[naive_p, nap]
        pci_p = np.c_[pci_p, sep]

_,col_num = np.shape(naive_p)

# Naive
wb = Workbook()
ws = wb.add_sheet("Whole list of naive p-value")
header2 = ["Gene Name"] + header1
for i in range(len(header2)):
    ws.write( 0, i, header2[i] )
for j in range(d):
    ws.write( j+1, 0, gene_name[j] )
    for i in range(col_num):
        ws.write( j+1, i+1, naive_p[j,i] )    
wb.save("{}_whole_pval_naive.xls".format(dataname))

# scPCI
wb = Workbook()
ws = wb.add_sheet("Whole list of scPCI p-value")
for i in range(len(header2)):
    ws.write( 0, i, header2[i] )
for j in range(d):
    ws.write( j+1, 0, gene_name[j] )
    for i in range(col_num):
        ws.write( j+1, i+1, pci_p[j,i] )    
wb.save("{}_whole_pval_scPCI.xls".format(dataname))

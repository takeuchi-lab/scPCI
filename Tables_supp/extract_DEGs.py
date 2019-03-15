# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:33:16 2018

@author: Inoue.S
"""

import numpy as np

dataname = "FACSfat" #PBMC, FACSfat

# Significant level
alpha = 0.01

if dataname == "PBMC":
    K = 10
    dir_name = "../PCI_PBMC/"
    _path = dir_name + "each/logPBMC_K{}".format(K)
    gene_name = np.load(dir_name + "use_gene_name.npy")

if dataname == "FACSfat":
    K = 5
    dir_name = "../PCI_FACSfat/"
    _path = dir_name + "each/logFACSfat_K{}".format(K)
    gene_name = np.load(dir_name + "use_gene_name_Fat.npy")

d = len(gene_name)

# If you set numRej_lower=K-1 and num_celltype  = 1, then you can get only one cell-type DEGs.
# If you set numRej_lower=K-1 and num_celltype >= 2, then you can get multi cell-type DEGs.
numRej_lower = K-1
num_celltype = K

def extact_DEGs(K, path, gene_name, d, alpha, numRej_lower=K-1, num_celltype=K):
    # bonferroni correction (option)
    adjust = (K * (K-1) / 2.) * d
    
    # load result
    table_naive = np.array( [ np.infty * np.ones((K,K)) ] * d )
    table_pci = np.array( [ np.infty * np.ones((K,K)) ] * d )
    for a in range(K):
        li = list( range(K) )
        li.remove(a)
        for b in li:
            if a > b:
                _fn = _path + "_a{}b{}.npz".format(b+1,a+1)
            else:
                _fn = _path + "_a{}b{}.npz".format(a+1,b+1)
            tmp = np.load(_fn)
            nap = tmp["nap"]
            table_naive[:,a,b] = np.min(np.c_[adjust*nap, np.ones(d)], axis=1)
            sep = tmp["sep"]
            table_pci[:,a,b] = np.min(np.c_[adjust*sep, np.ones(d)], axis=1)
    
    # Number of rejection for each cluster 
    numRej_naive = np.sum(table_naive < alpha, axis=2)
    numRej_pci   = np.sum(table_pci   < alpha, axis=2)
    
    # Flag for (numRej_hoge >= numRej_lower)
    speFlag_naive = ( numRej_naive >= numRej_lower )
    speFlag_pci   = ( numRej_pci   >= numRej_lower )
    sum_naive1 = np.sum(speFlag_naive, axis=1)
    sum_pci1 = np.sum(speFlag_pci, axis=1)
    
    # Gene flags for (numRej_hoge >= numRej_lower)
    genefg_naive = (0 < sum_naive1) * (sum_naive1 <= num_celltype)
    genefg_pci   = (0 < sum_pci1)   * (sum_pci1   <= num_celltype)
    
    # Extract DEGs
    speGenes_naive = [ [] for k in range(K) ]
    speGenes_pci   = [ [] for k in range(K) ]
    for j in range(d):
        gn = gene_name[j]
        if genefg_naive[j] == True:
            loc1 = speFlag_naive[j].nonzero()[0]
            for pos in loc1:
                speGenes_naive[pos].append(gn)
        if genefg_pci[j] == True:
            loc2 = speFlag_pci[j].nonzero()[0]
            for pos in loc2:
                speGenes_pci[pos].append(gn)
    
    for k in range(K):
        tmp = speGenes_naive[k]
        tmp.sort()
        speGenes_naive[k] = ["cluster {}".format(k+1), len(tmp)] + speGenes_naive[k]
        tmp = speGenes_pci[k]
        tmp.sort()
        speGenes_pci[k] = ["cluster {}".format(k+1), len(tmp)] + speGenes_pci[k]
    
    # Output csv files
    import csv
    with open('naive_DEGs.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for k in range(K):
            writer.writerow(speGenes_naive[k])
    
    with open('scPCI_DEGs.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for k in range(K):
            writer.writerow(speGenes_pci[k])
    
    with open("setting.txt", "w") as f:
        f.write("Bonferroni correction. alpha={}. numRej_lower={}. num_celltype={}."
                .format(alpha, numRej_lower, num_celltype))
        
extact_DEGs(K, _path, gene_name, d, alpha, numRej_lower, num_celltype)

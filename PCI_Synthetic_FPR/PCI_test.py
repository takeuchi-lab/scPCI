# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 12:55:39 2017

@author: Inoue.S
"""

import numpy as np
from method_sub import make_data_quick, make_data, kmeans, PCI_ell1, PCI_ell2
import time
from progressbar import ProgressBar, Percentage, Bar

ite = 10**4
_date = "1.31/"

# No. of Cluster
K = 2
# 0: Global Null, 1: Hetero
_flag = 0
move_sample = 0
mu = 1

a = 0
b = 1

#np.random.seed(0)

if _flag==1:
    # sample & feature
    n = 100
    d = 100
    Sigma = np.identity(n)
    xi = 1
    Xi = xi*np.identity(d)
    _ratio = np.array([1, 5, 10, 20]) / 100.
    fn = "hetero_n{}_d{}_mu{}_K{}".format(n,d,mu,K)
    print(fn)
    _mulist = [mu]
    _nlist = [[0,50]]
    
    naive_pow1 = []
    naive_pow2 = []
    pci_pow1 = []
    pci_pow2 = []
    for _r in _ratio:
        q = int(d*_r)
        _qlist = [[0,q]]
        print("ratio:  {}".format(_r))
        nap_list1 = []
        nap_list2 = []
        sep_list1 = []
        sep_list2 = []
        p = ProgressBar(widgets=[Percentage(), Bar()], maxval=ite).start()
        start = time.time()
        for _t in range(ite):
            p.update(_t+1)
            # make_data
            X = make_data_quick(n, d, _mulist, _nlist, _qlist, xi)
            # PCI ell_1 ver
            init = np.random.choice(range(n), K, replace=False)
            lvec_T,c_v = kmeans(X, K, init, "cityblock", T=100)
            _nap1,_sep1,_,_,_,_ = PCI_ell1(0, 1, init, lvec_T, Sigma, Xi, X, K, T=100)
            # PCI ell_2 ver
            init = np.random.choice(range(n), K, replace=False)
            lvec_T,c_v = kmeans(X, K, init, "sqeuclidean", T=100)
            _nap2,_sep2,_,_,_ = PCI_ell2(0, 1, init, lvec_T, Sigma, Xi, X, K, T=100)
            nap_list1.append(_nap1)
            nap_list2.append(_nap2)
            sep_list1.append(_sep1)
            sep_list2.append(_sep2)
        print("Time: {}".format(time.time() - start))
        p.finish()
        nap_list1 = np.array(nap_list1)
        nap_list2 = np.array(nap_list2)
        sep_list1 = np.array(sep_list1)
        sep_list2 = np.array(sep_list2)
        naive_pow1.append(np.mean(nap_list1<0.05, axis=0))
        naive_pow2.append(np.mean(nap_list2<0.05, axis=0))
        pci_pow1.append(np.mean(sep_list1<0.05, axis=0))
        pci_pow2.append(np.mean(sep_list2<0.05, axis=0))
    naive_pow1 = np.array(naive_pow1)
    naive_pow2 = np.array(naive_pow2)
    pci_pow1 = np.array(pci_pow1)
    pci_pow2 = np.array(pci_pow2)
    np.savez(_date+fn, naive_pow1 = naive_pow1, naive_pow2 = naive_pow2,
             pci_pow1 = pci_pow1, pci_pow2 = pci_pow2)
    
else:
    print("Data is global null")
    if move_sample > 0:
        d = 10
        xi = 1
        Xi = xi*np.identity(d)
        var_list = [50, 100, 200, 500]
    else: 
        n = 100
        Sigma = np.identity(n)
        var_list = [10, 20, 50, 100]
    _mulist = []
    _nlist = []
    _qlist = []
    naive_fpr1 = []
    naive_fpr2 = []
    pci_fpr1 = []
    pci_fpr2 = []
    for v in var_list:
        if move_sample > 0:
            n = v
            Sigma = np.identity(n)
        else: 
            d = v
            xi = 1
            Xi = xi*np.identity(d)
        fn = "gn_n{}_d{}_K{}".format(n,d,K)
        print(fn)
        nap_list1 = []
        nap_list2 = []
        sep_list1 = []
        sep_list2 = []
        p = ProgressBar(widgets=[Percentage(), Bar()], maxval=ite).start()
        start = time.time()
        for _t in range(ite):
            p.update(_t+1)
            # make_data
            X = make_data_quick(n, d, _mulist, _nlist, _qlist, xi)
            # PCI ell_1 ver
            init = np.random.choice(range(n), K, replace=False)
            lvec_T,c_v = kmeans(X, K, init, "cityblock", T=100)
            _nap1,_sep1,_,_,_,_ = PCI_ell1(0, 1, init, lvec_T, Sigma, Xi, X, K, T=100)
            # PCI ell_2 ver
            init = np.random.choice(range(n), K, replace=False)
            lvec_T,c_v = kmeans(X, K, init, "sqeuclidean", T=100)
            _nap2,_sep2,_,_,_ = PCI_ell2(0, 1, init, lvec_T, Sigma, Xi, X, K, T=100)
            nap_list1.append(_nap1)
            nap_list2.append(_nap2)
            sep_list1.append(_sep1)
            sep_list2.append(_sep2)
        print("Time: {}".format(time.time() - start))
        p.finish()
        nap_list1 = np.array(nap_list1)
        nap_list2 = np.array(nap_list2)
        sep_list1 = np.array(sep_list1)
        sep_list2 = np.array(sep_list2)
        naive_fpr1.append(np.mean(nap_list1<0.05, axis=0)[0])
        naive_fpr2.append(np.mean(nap_list2<0.05, axis=0)[0])
        pci_fpr1.append(np.mean(sep_list1<0.05, axis=0)[0])
        pci_fpr2.append(np.mean(sep_list2<0.05, axis=0)[0])
    naive_fpr1 = np.array(naive_fpr1)
    naive_fpr2 = np.array(naive_fpr2)
    pci_fpr1 = np.array(pci_fpr1)
    pci_fpr2 = np.array(pci_fpr2)
    np.savez(_date+fn, naive_fpr1 = naive_fpr1, naive_fpr2 = naive_fpr2,
             pci_fpr1 = pci_fpr1, pci_fpr2 = pci_fpr2)
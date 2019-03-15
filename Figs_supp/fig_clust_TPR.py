# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:06:13 2019

@author: Inoue.S
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

n = 50
d_li = [2, 10, 50]
mu = 2.0
str_mu = str(mu).replace(".","p")

dir_name = "../PCIgn_Synthetic_TPR/"

fs = 20
x = np.arange(0.50, 1.01, 0.01)
for d in d_li:
    tpr = np.load(dir_name + "hetero_n{}d{}mu{}_gn.npy".format(n,d,mu))
    tpr[:,0] = 1 - tpr[:,0]
    
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(tpr[:,0].reshape(-1,1), tpr[:,1])
    pred = lr.predict_proba(x.reshape(-1,1))
    
    # Plot
    fig, ax = plt.subplots(nrows=2, figsize=(8,8), sharex=True)
    plt.subplot(2,1,1)
    plt.ylabel("Estimated TPR",fontsize=fs)
    plt.plot(x, pred[:,1], label="scPCI-cluster ($d={})$".format(d), linewidth=2)
    plt.ylim([0,1])
    plt.legend(loc="best", fontsize=fs)
    plt.tick_params(labelsize=fs)

    rej = tpr[tpr[:,1] == 1 ,0]
    acc = tpr[tpr[:,1] == 0 ,0]
    plt.subplot(2,1,2)    
    plt.xlabel("Accuracy of clustering (Jaccard index)",fontsize=fs)
    plt.ylabel("Frequency",fontsize=fs)
    plt.hist([rej, acc], stacked=True, normed = True, label=["Positive", "Negative"], 
             bins=20, range=(0.5,1.0), color=("red","k"), alpha=0.8)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs)
    plt.savefig("hetero_n{}d{}mu{}_gn".format(n,d,str_mu)+".pdf", bbox_inches="tight")
    plt.show()
    
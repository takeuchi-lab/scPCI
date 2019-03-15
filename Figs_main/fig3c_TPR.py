# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:53:17 2018

@author: Inoue.S
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

n = 50
#d_li = [2, 10, 50]
d_li = [2, 50]
mu = 1.0 #paper -> 1.0,2.0
str_mu = str(mu).replace(".","p")

dir_name = "../PCI_Synthetic_TPR/"

sns.set_style("white")
fs = 20
x = np.arange(0.50, 1.01, 0.01)
for d in d_li:
    fn = dir_name + "hetero_n{}d{}mu{}_ex4".format(n,d,mu)
    tmp = np.load(fn+".npz")
    tpr = tmp["tpr"]
    # Trandsform EMdist to Jaccard index
    tpr[:,0] = 1 - tpr[:,0]
    
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(tpr[:,0].reshape(-1,1), tpr[:,1])
    pred = lr.predict_proba(x.reshape(-1,1))
    
#    # Naive's Power
#    z_alpha = stats.norm.isf(0.05)
#    pow_naive = 1 + stats.norm.cdf(-z_alpha-x) - stats.norm.cdf(z_alpha-x)
        
    # Plot
    fig, ax = plt.subplots(nrows=2, figsize=(8,8), sharex=True)
    plt.subplot(2,1,1)
    plt.ylabel("Estimated TPR",fontsize=fs)
    plt.plot(x, pred[:,1], label="scPCI $d={}$".format(d), linewidth=2)
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
    #plt.savefig("hetero_n{}d{}mu{}".format(n,d,str_mu)+".pdf", bbox_inches="tight", pad_inches=0.0)
    plt.savefig("hetero_n{}d{}mu{}".format(n,d,str_mu)+".jpeg", 
                bbox_inches="tight", dpi=1600)
    plt.show()
    
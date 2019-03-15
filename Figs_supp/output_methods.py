# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:41:41 2017

@author: Inoue.S
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

###############################################################################
##### Visualize clustering results
###############################################################################
# FACSfat: visualize by using tSNE
def visualize_FACSfat(c_result, index2, size):
    df_fat = pd.read_csv("../PCI_FACSfat/tSNE_data_Fat.csv", sep=",").loc[index2,:]
    tSNE_x = df_fat["tSNE_1"].values
    tSNE_y = df_fat["tSNE_2"].values

    # Result figure
    text_li = ["1. T,NK (471)","2. myeloid cell (1030)",
               "3. mesenchymal stem cell \n of adipose (1888)",
               "4. endothelial cell (596)", "5. B (524)"]
    locx = [-55, 15,  20, -48, -25]
    locy = [ 12, 22, -35,  28, -18]
    plt.figure(figsize=(13,10))
    c2 = ["red","#1c86ee","#8B008B","#6e8b3d","#99c9fb"]
    for i in range(5):
        fg = ( np.array(c_result)==i )
        plt.scatter(tSNE_x[fg], tSNE_y[fg], s=size, c=c2[i])
        plt.text(locx[i], locy[i], text_li[i], fontsize=28, color=c2[i])
    # -------------------
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.savefig("clust_FACSfat_result.jpg", bbox_inches="tight", dpi=1600)
    plt.show()

###############################################################################
##### visualize expr for each gene
###############################################################################
# PBMC: visualize by using tSNE for each gene
def visualize_expr(expr, gene_name, num, size):
    n = len(expr)
    sub = np.sort( np.random.choice(range(n), num, replace=False) )
        
    expr = expr[sub]
    result = np.load("../PCI_PBMC/result_tsne.npy")
    result = result[sub,:]
    # colormap: seismic, OrRd, RdPu
    plt.figure()
    plt.scatter(result[:,0], result[:,1], s=size, c=expr, cmap="RdPu", alpha=0.8)
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.title(gene_name, fontsize=18)
    plt.xlabel("tSNE1", fontsize=18)
    plt.ylabel("tSNE2",fontsize=18)
    plt.colorbar()
    plt.savefig("expr_{}.jpg".format(gene_name), bbox_inches="tight", dpi=1600)
    plt.show()

# FACSfat: visualize by using tSNE for each gene
def visualize_expr_FACSfat(expr, gene_name, size):
    df_fat = pd.read_csv("../PCI_FACSfat/tSNE_data_Fat.csv", sep=",")
    tSNE_x = df_fat["tSNE_1"].values
    tSNE_y = df_fat["tSNE_2"].values
    
    plt.figure()
    plt.scatter(tSNE_x, tSNE_y, s=size, c=expr, cmap="RdPu", alpha=0.8)
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.title(gene_name, fontsize=18)
    plt.xlabel("tSNE1", fontsize=18)
    plt.ylabel("tSNE2",fontsize=18)
    plt.colorbar()
    plt.savefig("expr_{}.jpg".format(gene_name), bbox_inches="tight", dpi=1600)
    plt.show()

###############################################################################
##### p-value heatmaps
###############################################################################
def draw_heatmap2(fn, X1, X2, xtickls, ytickls, gn):
    colors = ["white"]*1 + ["lightgreen"]*4 + ["yellowgreen"]*5 + ["green"]*90
    my_cm = LinearSegmentedColormap.from_list("my_list", colors, 100)
    plt.title("naive : {}".format(gn), fontsize=20)
    sns.set(font_scale=1.5)
    sns.heatmap(X1, cmap=my_cm, annot=True, linewidths=1, square=True, 
                linecolor='white', cbar=False, fmt="1.3f", annot_kws={"size": 12}, 
                cbar_kws=None, cbar_ax=None, vmax=1.0, vmin=0.0,
                xticklabels=xtickls, yticklabels=ytickls)
    plt.xlabel("Cluster index", fontsize=20)
    plt.ylabel("Cluster index", fontsize=20)
    plt.savefig(fn+"_naive.jpeg", bbox_inches="tight", dpi=1600)
    plt.show()

    plt.title("scPCI : {}".format(gn), fontsize=20)
    sns.set(font_scale=1.5)
    sns.heatmap(X2, cmap=my_cm, annot=True, linewidths=1, square=True, 
                linecolor='white', cbar=False, fmt="1.3f", annot_kws={"size": 12}, 
                cbar_kws=None, cbar_ax=None, vmax=1.0, vmin=0.0, 
                xticklabels=xtickls, yticklabels=ytickls)
    plt.xlabel("Cluster index", fontsize=20)
    plt.ylabel("Cluster index", fontsize=20)
    plt.savefig(fn+"_pci.jpeg", bbox_inches="tight", dpi=1600)
    plt.show()

# p-value heatmap for clustering
def draw_heatmap_gn(fn, X1, X2, xtickls, ytickls):
    colors = ["white"]*1 + ["deepskyblue"]*4 + ["blue"]*5 + ["navy"]*90
    my_cm = LinearSegmentedColormap.from_list("my_list", colors, 100)
    plt.title("naive $\chi^2$", fontsize=20)
    sns.set(font_scale=1.5)
    sns.heatmap(X1, cmap=my_cm, annot=True, linewidths=1, square=True, 
                linecolor='white', cbar=False, fmt="1.3f", annot_kws={"size": 12}, 
                cbar_kws=None, cbar_ax=None, vmax=1.0, vmin=0.0,
                xticklabels=xtickls, yticklabels=ytickls)
    plt.xlabel("Cluster index", fontsize=20)
    plt.ylabel("Cluster index", fontsize=20)
    plt.savefig(fn+"_naive.jpeg", bbox_inches="tight", dpi=1600)
    plt.show()
    
    plt.title("scPCI-cluster", fontsize=20)
    sns.set(font_scale=1.5)
    sns.heatmap(X2, cmap=my_cm, annot=True, linewidths=1, square=True, 
                linecolor='white', cbar=False, fmt="1.3f", annot_kws={"size": 12}, 
                cbar_kws=None, cbar_ax=None, vmax=1.0, vmin=0.0, 
                xticklabels=xtickls, yticklabels=ytickls)
    plt.xlabel("Cluster index", fontsize=20)
    plt.ylabel("Cluster index", fontsize=20)
    plt.savefig(fn+"_pci.jpeg", bbox_inches="tight", dpi=1600)
    plt.show()

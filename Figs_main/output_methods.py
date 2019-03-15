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
# PBMC: visualize by using tSNE
def visualize(c_result, index2, num, size):
    n = len(index2)
    sub = np.random.choice(range(n), num, replace=False)
    c_result = (np.array(c_result) + 1)[sub]
    result = np.load("../PCI_PBMC/result_tsne.npy")
    result = (result[index2,:])[sub,:]
    c = ["#FB9A99","saddlebrown","yellowgreen","orchid","grey","red","#1c86ee",
         "#8B008B","#6e8b3d","#99c9fb"]
    # Result figure
    text_li = ["1. Activated CD8+ (6524)","2. Naive CD8+ (13684)","3. Memory and RegT (11881)",
               "4. Naive CD4+ (14240)", "5. NK (4136)", "6. CD8+ (8312)", "7. B (3821)",
               "8. Magakaryocytes (161)","9. Monocytes,dendritic (3843)","10. B,dendritic,T (977)"]
    locx = [23, -58, -55, -18,  12, 25, 10,  22,  30, -30]
    locy = [20, - 5,  15, -30, -25, 10, 30, -18, - 5,  30]
    plt.figure(figsize=(13,10))
    for i in range(1,11):
        fg = (c_result==i)
        plt.scatter(result[fg,0], result[fg,1], s=size, c=c[i-1])
        plt.text(locx[i-1], locy[i-1], text_li[i-1], fontsize=28, color=c[i-1])
    # --------- 以下追加部分----------
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.quiver(25, -15, -3, 9.5, angles='xy', scale_units='xy', scale=1, width=0.004)
    plt.quiver(25, 10, -13, -5, angles='xy', scale_units='xy', scale=1, width=0.004)
    plt.savefig("clust_PBMC_result.jpeg", bbox_inches="tight", dpi=1600)
    plt.show()

# FACSfat: visualize by using tSNE
def visualize_FACSfat(c_result, index2, size):
    df_fat = pd.read_csv("PCI_FACSfat/tSNE_data_Fat.csv", sep=",").loc[index2,:]
    #cell_type = df_fat["cell_ontology_class"]
    #label = list( set(cell_type) )
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
    # --------- 以下追加部分----------
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.savefig("pdf/clust_FACSfat_result.png", bbox_inches="tight", dpi=200)
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
    plt.savefig("expr_{}.jpeg".format(gene_name), bbox_inches="tight", dpi=1600)
    plt.show()

# FACSfat: visualize by using tSNE for each gene
def visualize_expr_FACSfat(expr, gene_name, size):
    df_fat = pd.read_csv("PCI_FACSfat/tSNE_data_Fat.csv", sep=",")
    tSNE_x = df_fat["tSNE_1"].values
    tSNE_y = df_fat["tSNE_2"].values
    
    plt.figure(figsize=(int(1.25*h),h))
    plt.scatter(tSNE_x, tSNE_y, s=size, c=expr, cmap="RdPu", alpha=0.8)
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.title(gene_name, fontsize=18)
    plt.xlabel("tSNE1", fontsize=18)
    plt.ylabel("tSNE2",fontsize=18)
    plt.colorbar()
    plt.savefig("pdf/expr_{}.png".format(gene_name), bbox_inches="tight")
    plt.show()

###############################################################################
##### p-value heatmaps
###############################################################################
def draw_heatmap2(fn, X1, X2, clr, xtickls, ytickls, gn):
    colors = ["white"]*1 + ["lightgreen"]*4 + ["yellowgreen"]*5 + ["green"]*90
    my_cm = LinearSegmentedColormap.from_list("my_list", colors, 100)
    plt.title("naive : {}".format(gn), fontsize=20)
    sns.set(font_scale=1.5)
    sns.heatmap(X1, cmap=my_cm, annot=True, linewidths=1, square=True, 
                linecolor='black', cbar=False, fmt="1.3f", annot_kws={"size": 12}, 
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
def draw_heatmap_gn(fn, X1, X2, clr, xtickls, ytickls):
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

###############################################################################
##### Others
###############################################################################
def prt_Tex_format(name, dim, arr):
    tformat = "{} & ".format(name)
    string_fg = type(arr[0]) is np.str_
    for i in range(dim-1):
        if string_fg:
           tformat += "{} & ".format(arr[i]) 
        else:
           tformat += "{:.3g} & ".format(arr[i])
    if string_fg:
        tformat += "{}".format(arr[i+1])
    else:
        tformat += "{:.3g}".format(arr[i+1])
    print(tformat + " \\\\ \hline")

def prt_Tex_table(title, col_name, arr):
    n,d = np.shape(arr)
    for i in range(n):
        prt_Tex_format(col_name[i], d, arr[i])
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\caption{" + title + "}")
    print("\\end{table}")

def prt_top_signif_genes(title, num_top, gene_name, sep, nap, T, naive_order=True):
    if naive_order: # correspond to Naive order
        st_arg = np.argsort(np.abs(T))[::-1]
    else: # correspond to PCI order
        st_arg = np.argsort(sep)
    q1 = gene_name[st_arg][:num_top]
    print("\\begin{table}[htbp]")
    print("\\begin{center}")
    tmp = "|" + "c|"*(num_top+1)
    print("\\begin{tabular}{" + "{}".format(tmp) + "}")
    prt_Tex_format("Gene", num_top, q1)
    q2 = sep[st_arg][:num_top] # PCI
    q3 = nap[st_arg][:num_top] # Naive
    q4 = np.abs(T)[st_arg][:num_top] # Naive
    arr = np.c_[q2,q3,q4].T
    prt_Tex_table(title, ["PCI", "Naive", "Stats"], arr)

def pie(lvec_T, K):
    cT = [ lvec_T.count(k) for k in range(K) ]
    n = sum(cT)
    sizes = np.array(cT) / n
    labels = np.array(range(1,11))
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f', startangle=90, pctdistance=1.2, labeldistance=0.7)
    ax1.axis('equal')
    plt.savefig("pdf/logPBMC_pie.pdf")
    plt.show()

def visualize_forkl(c_result, outl, inl, index2, size):
    c_result = (np.array(c_result) + 1)
    result = np.load("171122/result_tsne.npy")
    result = result[index2,:]
    in_result = result[inl]
    out_result = result[outl]
    label = ["Activated CD8+","Naive CD8+","Memory and RegT","Naive CD4+",
             "NK","CD8+","B","Magakaryocytes","Monocytes and dendritic","B, dendritic, T"]
    label1 = range(1,11)
    c = ["#FB9A99","#FF7F00","yellow","orchid","grey","red","#1c86ee","#8B008B","#6e8b3d","#99c9fb"]
    
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(22,10))
    for i in range(1,11):
        fg = (c_result==i)
        axL.scatter(in_result[fg,0], in_result[fg,1], s=size, c=c[i-1], label=label1[i-1])
        fg = (result[:,2]==i)
        axR.scatter(result[fg,0], result[fg,1], s=size, c=c[i-1], label=label[i-1])
    axR.legend(fontsize=size)
    axL.legend(fontsize=size)
    axL.scatter(out_result[:,0], out_result[:,1], s=size, c="k", marker="^")
    axL.set_title('Result', fontsize=18)
    axR.set_title('Reference', fontsize=18)
    fig.savefig("pdf/clustering_result.pdf")
    fig.show()

def check_corr(num_top, sep, T, naive_order=True):
    if naive_order: # correspond to Naive order
        st_arg = np.argsort(np.abs(T))[::-1]
    else: # correspond to PCI order
        st_arg = np.argsort(sep)
    q1 = sep[st_arg] # PCI
    q2 = np.abs(T)[st_arg] # Naive
    
    st_arg1 = np.argsort(q1[:num_top])+1
    st_arg2 = np.argsort(q2[:num_top])[::-1]+1
    corr = np.corrcoef(st_arg1, st_arg2)
    return corr

def check_signif(num_top, sep, nap, gene_name, naive_order=True):
    if naive_order: # correspond to Naive order
        st_arg = np.argsort(np.abs(T))[::-1]
    else: # correspond to PCI order
        st_arg = np.argsort(sep)
    st_gn = gene_name[st_arg]
    q1 = sep[st_arg] # PCI
    q2 = np.abs(T)[st_arg] # Naive
    
    x =  np.arange(num_top)
    ya = q2[:num_top]
    yb = -np.log10( q1[:num_top] )
    width = 0.25
    plt.ylabel("$-\log_{10}(p)$")
    plt.bar(x, ya, width)
    plt.bar(x + width, yb, width)
    plt.title("Naive top {} genes".format(num_top))
    plt.xticks(x, st_gn[:num_top], rotation=70)
    plt.show()

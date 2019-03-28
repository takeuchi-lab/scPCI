#
# Copyright (c) 2016 10x Genomics, Inc. All rights reserved.
#

rm(list=ls()) # clear workspace
# ----------------------------
# load relevant libraries
# ----------------------------
library(Matrix)
library(ggplot2)
#library(Rtsne)
library(svd)
library(plyr)
library(dplyr)
#library(data.table)
#library(pheatmap)
# -------------------------------------
# specify paths and load functions
# -------------------------------------
DATA_DIR <- "./"        # SPECIFY HERE
PROG_DIR <- "./"     # SPECIFY HERE
RES_DIR  <- "./"      # SPECIFY HERE
source(file.path(PROG_DIR,'util.R')) 
# ------------------------------------------------------------
# load 68k PBMC data, 11 purified PBMC data and meta-data
# ------------------------------------------------------------
pbmc_68k <- readRDS(file.path(DATA_DIR,'pbmc68k_data.rds'))
pure_11 <- readRDS(file.path(DATA_DIR,'all_pure_select_11types.rds'))
all_data <- pbmc_68k$all_data
purified_ref_11 <- load_purified_pbmc_types(pure_11,pbmc_68k$ens_genes)
# --------------------------------------------------------------------------------------
# normalize by RNA content (umi counts) and select the top 1000 most variable genes
# --------------------------------------------------------------------------------------
m<-all_data[[1]]$hg19$mat
l<-.normalize_by_umi(m)   
m_n<-l$m
df<-.get_variable_gene(m_n) 
disp_cut_off<-sort(df$dispersion_norm,decreasing=T)[1000]
df$used<-df$dispersion_norm >= disp_cut_off
# --------------------------------------------------
# plot dispersion vs. mean for the genes
# this produces Supp. Fig. 5c in the manuscript
# --------------------------------------------------
# ggplot(df,aes(mean,dispersion,col=used))+geom_point(size=0.5)+scale_x_log10()+scale_y_log10()+
  scale_color_manual(values=c("grey","black"))+theme_classic()
# --------------------------------------------
# use top 1000 variable genes for PCA 
# --------------------------------------------
m_n_1000<-m_n[,head(order(-df$dispersion_norm),1000)]
denceX <- as.matrix(m_n_1000)
write.csv(denceX, "processX.csv", row.names=FALSE, col.names=FALSE)
all_gene_name <- all_data[[1]]$hg19$gene_symbols
gene_name <- all_gene_name[l$use_genes]
gene_name <- gene_name[head(order(-df$dispersion_norm),1000)]
write.csv(gene_name, "gene_name.csv", row.names=FALSE, col.names=FALSE)

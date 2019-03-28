
setwd("./tabula-muris-master/00_data_ingest")

library(Seurat)
library(dplyr)
library(Matrix)
library(stringr)
library(here)

Load the count data for one organ and add it to the Seurat object.

organ = "Fat"

raw.data <- read.csv(here("00_facs_raw_data", "FACS",paste0(organ,"-counts.csv")),row.names = 1)
#raw.data <- Matrix(as.matrix(raw.data), sparse = TRUE)
meta.data <- read.csv(here("00_facs_raw_data", "metadata_FACS.csv"))

plates <- str_split(colnames(raw.data),"[.]", simplify = TRUE)[,2]

rownames(meta.data) <- meta.data$plate.barcode
cell.meta.data <- meta.data[plates,]
rownames(cell.meta.data) <- colnames(raw.data)

# Find ERCC's, compute the percent ERCC, and drop them from the raw data.
erccs <- grep(pattern = "^ERCC-", x = rownames(x = raw.data), value = TRUE)
percent.ercc <- Matrix::colSums(raw.data[erccs, ])/Matrix::colSums(raw.data)
ercc.index <- grep(pattern = "^ERCC-", x = rownames(x = raw.data), value = FALSE)
raw.data <- raw.data[-ercc.index,]

# Create the Seurat object with all the data
tiss <- CreateSeuratObject(raw.data = raw.data, min.cells = 5, min.genes = 5)

tiss <- AddMetaData(object = tiss, cell.meta.data)
tiss <- AddMetaData(object = tiss, percent.ercc, col.name = "percent.ercc")
# Change default name for sums of counts from nUMI to nReads
colnames(tiss@meta.data)[colnames(tiss@meta.data) == 'nUMI'] <- 'nReads'

ribo.genes <- grep(pattern = "^Rp[sl][[:digit:]]", x = rownames(x = tiss@data), value = TRUE)
percent.ribo <- Matrix::colSums(tiss@raw.data[ribo.genes, ])/Matrix::colSums(tiss@raw.data)
tiss <- AddMetaData(object = tiss, metadata = percent.ribo, col.name = "percent.ribo")

percent.Rn45s <- Matrix::colSums(tiss@raw.data[c('Rn45s'), ])/Matrix::colSums(tiss@raw.data)
tiss <- AddMetaData(object = tiss, metadata = percent.Rn45s, col.name = "percent.Rn45s")

# A sanity check: genes per cell vs reads per cell.

GenePlot(object = tiss, gene1 = "nReads", gene2 = "nGene", use.raw=T)

# Filter out cells with few reads and few genes.
tiss <- FilterCells(object = tiss, subset.names = c("nGene", "nReads"), 
    low.thresholds = c(500, 50000), high.thresholds = c(25000, 2000000))


# Normalize the data, then regress out correlation with total reads
tiss <- NormalizeData(object = tiss)
tiss <- ScaleData(object = tiss, vars.to.regress = c("nReads", "percent.ribo","Rn45s"))
# tiss <- FindVariableGenes(object = tiss, do.plot = TRUE, x.high.cutoff = Inf, y.cutoff = 0.5)
# top 1000 variable genes
tiss <- FindVariableGenes(object = tiss, do.plot = TRUE, selection.method = "dispersion", top.genes=1000)
genes.use <- tiss@var.genes
# genes.var <- apply(X = tiss@data[genes.use, ], MARGIN = 1, FUN = var)
write.csv(gene.use, "use_gene_name_Fat.csv")
data.use <- tiss@data[genes.use,]
write.csv(as.data.frame(as.matrix(data.use)), "LogNorm_processX_Fat.csv")

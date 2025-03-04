# Load the necessary libraries
#library(affy)
library(oligo)
library(dplyr)
library(readr)
#library(AnnotationDbi)
#library(rat2302.db) # version 3.13.0 from Bioconductor 3.19 release
library(preprocessCore)
library(tibble)

# Set working directory and output directory
setwd('/path/to/data/celLiver_rat/celSingle')  # Replace with the actual path to your CEL files
output_dir <- '/path/to/rma_results/rat_vivo_liver_single'  # Replace with the desired output directory

# Normalization
cel.path <- '/path/to/data/celLiver_rat/celSingle'  # Replace with the actual path to your CEL files

# List all CEL files in the directory
all_cell_files <- list.files(cel.path, pattern='*.CEL')

# Perform RMA normalization
CEL <- read.celfiles(all_cell_files)
RMA <- rma(CEL)
saveRDS(object = RMA, file = file.path(output_dir, 'RMA.rds'))

# Extract expression matrix
EXPRS <- exprs(RMA)
saveRDS(object = EXPRS, file = file.path(output_dir, 'EXPRS.rds'))

# Read the expression matrix
EXPRS <- readRDS(file = file.path(output_dir, 'EXPRS.rds'))
DATA <- rownames_to_column(as.data.frame(EXPRS), var = "probeset_ID")

# Write the expression data to a TSV file
write_tsv(DATA, file = file.path(output_dir, "EXPRS.tsv"), col_names = TRUE)

# The EXPRS.tsv can be converted to a CSV and is the normalized data (vivo_norm.csv)

# Clean up workspace
rm(CEL, RMA, EXPRS, DATA)
gc()

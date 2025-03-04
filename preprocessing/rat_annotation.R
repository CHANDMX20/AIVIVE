# Load necessary libraries
library(affy)
library(dplyr)
library(readr)
library(AnnotationDbi)
library(rat2302.db) # version 3.13.0 from Bioconductor 3.19 release
library(preprocessCore)
library(tibble)

# Read the normalized CSV file for rats
rat_norm = read.csv("/path/to/vivo_norm.csv")  # Replace with your actual file path

# Extract probe IDs
rat_probe_ids <- rat_norm$probeset_ID

# Download the annotation file for rat
rat2302 <- rat2302.db

# Extract the annotation data for rat
rat_annotation_data <- select(rat2302, keys = rat_probe_ids, columns = c("PROBEID", "GENENAME", "SYMBOL"))

# Read rat_s1500 dataset
rat_s1500 = read.csv("/path/to/rat_s1500.csv")  # Replace with your actual file path
colnames(rat_s1500) <- gsub("[. ]", "_", colnames(rat_s1500))  # Clean column names

# Rename columns in rat_s1500 to match annotation data
colnames(rat_s1500)[which(colnames(rat_s1500) == "Gene_Symbol")] <- "SYMBOL"

# Extract unique genes from the rat_s1500 dataset
rat_unique_genes <- unique(rat_s1500$SYMBOL)

# Convert the vector of unique genes to a data frame
rat_unique_genes <- data.frame(SYMBOL = rat_unique_genes)

# Merge rat annotation data with unique genes data based on the SYMBOL column
rat_data <- rat_annotation_data %>%
  inner_join(rat_unique_genes, by = "SYMBOL", relationship = "many-to-many")

# Rename columns for clarity
colnames(rat_data)[colnames(rat_data) == "PROBEID.x"] <- "PROBEID"

# Select relevant columns for the final data
rat_final_data <- rat_data %>%
  select(PROBEID, GENENAME, SYMBOL)

# Save the final rat data to a CSV file
write.csv(rat_final_data, "/path/to/final_rat_genes.csv", row.names = FALSE)  # Replace with your desired file path

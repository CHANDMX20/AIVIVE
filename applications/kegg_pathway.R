# load the packages 
library(clusterProfiler)
library(org.Rn.eg.db)
library(tidyr)
library(readr)
library(purrr)
library(dplyr)
library(stringr)

# load the dataset (change as necessary)
deg <- read.csv('path/to/vivo_union_deg_0.58.csv')   #path to the generated or real DEGs file generated from real_deg.py or gen_union.py (generated profiles DEGs in this case)

# For real degs ONLY

# Function to process the gene list and return Entrez IDs
#process_genes <- function(original_string) {
  # Remove the curly braces and single quotes
#original_string <- gsub("^\\[|\\]$", "", original_string)
  #original_string <- gsub("[{}']", "", original_string)

    
  # Split the string by commas
  #split_string <- unlist(strsplit(original_string, ", "))
  
  # Check if the split string is empty
  #if (length(split_string) == 0) {
    return(NULL)  # Return NULL if no genes
  #}
  
  # Convert gene symbols to Entrez IDs
  #gene_entrez_ids <- bitr(split_string, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Rn.eg.db)
  
  #return(gene_entrez_ids)
#}

# Apply the function to each row in the 'DEGs_0.58' column to generate Entrez IDs; DEGs for 2_Drugs file 
#deg$ENTREZID <- apply(deg, 1, function(row) {
  #gene_entrez_ids <- process_genes(row[["DEGs_0.58"]])
  
  # Return Entrez IDs or NA if no Entrez IDs are found
  #if (!is.null(gene_entrez_ids)) {
    #return(gene_entrez_ids$ENTREZID)
  #} else {
    #return(NA)
  #}
#})

# For generated DEGs only

# Function to process the gene list and return Entrez IDs
process_genes <- function(original_string) {
  # Remove the curly braces and single quotes
  original_string <- gsub("[{}']", "", original_string)

    
  # Split the string by commas
  split_string <- unlist(strsplit(original_string, ", "))
  
  # Check if the split string is empty
  if (length(split_string) == 0) {
    return(NULL)  # Return NULL if no genes
  }
  
  # Convert gene symbols to Entrez IDs
  gene_entrez_ids <- bitr(split_string, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Rn.eg.db)
  
  return(gene_entrez_ids)
}

# Apply the function to each row in the 'DEGs_0.58' column to generate Entrez IDs
deg$ENTREZID <- apply(deg, 1, function(row) {
  gene_entrez_ids <- process_genes(row[["DEGs_0.58"]])
  
  # Return Entrez IDs or NA if no Entrez IDs are found
  if (!is.null(gene_entrez_ids)) {
    return(gene_entrez_ids$ENTREZID)
  } else {
    return(NA)
  }
})

# Assuming the ENTREZID column has been created in the deg DataFrame
# Initialize columns to store results
deg$kegg_descriptions <- NA
deg$kegg_pathway_count <- NA

# Loop through each row of the deg DataFrame
for (i in 1:nrow(deg)) {
  # Extract the Entrez IDs for the current row
  entrez_ids <- deg$ENTREZID[[i]]
  
  # Check if the Entrez IDs are not NA or empty
  if (!is.null(entrez_ids) && length(entrez_ids) > 0) {
    # Perform KEGG enrichment analysis
    kegg_enrichment <- enrichKEGG(gene = entrez_ids,
                                   organism = "rno",  # 'rno' is the code for Rattus norvegicus in KEGG
                                   pvalueCutoff = 0.05, pAdjustMethod = "bonferroni")
    
    # Check if the enrichment result is valid
    if (!is.null(kegg_enrichment) && nrow(kegg_enrichment) > 0) {
      # Store the descriptions and count of pathways
      deg$kegg_descriptions[i] <- paste(kegg_enrichment$Description, collapse = ", ")
      deg$kegg_pathway_count[i] <- nrow(kegg_enrichment)
    } else {
      # If no pathways found, store NA
      deg$kegg_descriptions[i] <- NA
      deg$kegg_pathway_count[i] <- 0
    }
  } else {
    # If no Entrez IDs, store NA
    deg$kegg_descriptions[i] <- NA
    deg$kegg_pathway_count[i] <- 0
  }
}
# Remove the specified columns after generating the pathways
deg$ENTREZID <- NULL
deg$DEG_Count <- NULL
deg$DEGs_0.58 <- NULL
deg$Gene_Count <- NULL
deg$DEGs <- NULL

# Save the modified dataframe with the new KEGG analysis results 
write.csv(deg, "post_pathways_r_0.05.csv", row.names = FALSE)

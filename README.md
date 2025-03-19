# AIVIVE

**AIVIVE** is a novel **Generative Adversarial Network (GAN)** framework that combines a **GAN-based translator** with local optimizers. It uses biologically meaningful **gene clusters (modules)** to facilitate the demonstration of ***in vitro-in vivo*** **extrapolation (IVIVE)**.


---

## Table of Contents

- [Introduction](#introduction)
- [Code Scripts](#code-scripts)
  - [Preprocessing](#preprocessing)
  - [AIVIVE Model Development, Training & Predictions](#aivive-model-development-training--predictions)
  - [DEG Analysis](#deg-analysis)
  - [KEGG Pathway Analysis](#kegg-pathway-analysis)
  - [AOP-Gene Expression Analysis](#aop-gene-expression-analysis)
  - [Necrosis Prediction Model](#necrosis-prediction-model)
  - [Model Evaluation](#model-evaluation)
- [Data Files](#data-files)
- [Installation](#installation)
- [License](#license)
  
---

## Introduction

This repository contains code files and models for training the AIVIVE framework and generating predictions for *in vitro-in vivo* extrapolation. Additionally, it provides scripts for applying the model to biologically relevant tasks, such as analyzing differentially expressed genes (DEGs), predicting necrosis, and other related applications.  

---

## Code Scripts

### **[Preprocessing](./preprocessing)**

This folder contains the code and scripts for data preprocessing, specifically focusing on **RMA (Robust Multi-array Average) normalization** and **annotation** of gene expression data. These preprocessing steps are critical for preparing data before training the **AIVIVE** model.

**Files**:
- [`rma_vivo_single.R`](./preprocessing/rma_vivo_single.R) - R script that performs **RMA normalization** on the raw gene expression data which is widely used for processing microarray data
- [`rat_annotation.R`](./preprocessing/rat_annotation.R) - R script processes rat gene expression data by extracting probe IDs and merging them with gene annotations (relevant gene identifiers) to generate a final dataset of unique rat genes used for AIVIVE model development.

These preprocessing steps ensure that the gene expression data is properly prepared for the downstream tasks of training the **AIVIVE** model.

---

### **[AIVIVE Model Development, Training & Predictions](./training)**

This folder contains the core code for developing and training the **AIVIVE** framework. It includes the implementation of a **GAN-based translator** model and local optimizers networks.

  - **GAN-Based Translator** framework for translating *in-vitro* transcriptomic profiles to *in-vivo* transcriptomic profiles.
  - **Local optimizers** for enhancing the model's performance.
  - **Predictions** for generating the test set predictions.

**Files**:
- [`vitro_vivo_GAN.py`](./training/vitro_vivo_GAN.py) - GAN-based translator framework script to train the **AIVIVE** model on the IVIVE dataset.
- [`train_test_samples.py`](./training/train_test_samples.py) - Generating test set predictions using the optimal generator from the GAN-based translator
- [`optim_neural_net_#.py`](./training/modules) - Local optimizer neural network frameworks for specific modules, where `#` refers to the module number (e.g., `optim_neural_net_18.py`, `optim_neural_net_20.py`, etc.). These scripts contain implementations for training different modules.
- [`module_test_evals.py`](./training/modules/module_test_evals.py) - Generating test set predicitons for specific modules using the optimal local optimizers.

---

### **[DEG Analysis](./degs)**

This folder contains the code and scripts related to **Differentially Expressed Genes (DEG)** analysis for both real and AIVIVE-generated synthetic profiles. The code is the same for both rat S1500+ and modules gene sets with different input files depending on the task.

  - **Foldchange Calculation**:  The Foldchange Calculation is the difference of gene expression between treatment and control condition. It is calulcated as: ***Fold Change = Treatment Expression - Control Expression***

  - **DEG Identification**: DEGs were defined as genes with an absolute fold change > 1 in both real and synthetic profiles. It is given by: ***DEG = |FoldChange| > 1***.
    
  - **DEG Overlap Calculation**: Measures the overlap between real and generated DEGs by comparing the intersection of both sets and normalizing by the total number of real DEGs.
It is calculated as: ***Overlap = (DEGs_real ∩ DEGs_generated) / DEGs_real***

**Files**:
- [`real_fold.py`](./degs/real_fold.py) - Foldchange calculation script for real profiles.
- [`gen_fold.py`](./degs/gen_fold.py) - Foldchange calculation script for generated profiles (to be used for both rat S1500+ and module genes with modifications in input).
- [`real_deg.py`](./degs/real_deg.py) - DEG identification script for real profiles.
- [`gen_union.py`](./degs/gen_union.py) - DEG identification script for generated profiles. For synthetic profiles, an additional step was taken to calculate the union of DEGs across the subgroups of a specific treatment group.  
- [`overlap_deg.py`](./degs/overlap_deg.py) - DEGs overlap calculation script.
  
---

### **[KEGG Pathway Analysis](./applications)**

This contains code for performing KEGG (Kyoto Encyclopedia of Genes and Genomes) pathway analysis, which is used to identify biological pathways enriched in differentially expressed genes (DEGs) for understanding underlying biological processes. 

  - **KEGG Pathway Enrichment Analysis**: This step would map genes to known pathways using the KEGG database.
  - **Pathway Overlap Calculation**: Calculates the overlap of KEGG pathways enriched in real and generated profiles DEGs.
  
- **Files**:
  - [`kegg_pathway.R`](./applications/kegg_pathway.R) - KEGG pathway enrichment analysis. 
  - [`overlap_pathways.py`](./applications/overlap_pathways.py) - Python script for calculating the pathway overlap between real and synthetic profiles.

---

### **[AOP-Gene Expression Analysis](./applications)**

This contains code for performing **AOP (Adverse Outcome Pathway)** analysis combined with **Gene Expression Analysis**. The goal is to investigate the percentage error in gene expression values between real and AIVIVE-generated synthetic test profiles for AOP-related genes.

  - **Percentage Error in Gene Expression**: The AOP genes were identified from the rat S1500+ gene set and percentage error was calulcated for the gene expression values between the real and AIVIVE-generated synthetic profiles. It was calculated as: ***% Error (Gene Expression) = [(Real - Synthetic)/Real] * 100***

- **Files**:
  - [`AOP.py`](./applications/AOP.py) - Python script for calculating the **percentage error** in gene expression values between real and **AIVIVE-generated** profiles, followed by **visualization** using a heatmap to identify patterns in gene expression between the two profiles.

---

### **[Necrosis Prediction Model](./applications)**

This contains code for modeling **necrosis** gene expression predictions. 

  - **Necrosis Predictive Model Development** - Model trained on real *in vivo* train profiles
  - **Necrosis Model Evaluation** - The model was evaluated on two distinct test sets: one consisting of real *in vivo* profiles and another consisting of optimized synthetic *in vivo* profiles generated by AIVIVE. 

- **Files**:
  - [`necrosis_prediction.py`](./applications/necrosis_prediction.py) - Python script for developing and evaluating the necrosis predictive model.
    
---

### **[Model Evaluation](./applications)**

This contains the scripts for performing AIVIVE model evaluation on the test set using three metrices: cosine similarity, root mean squared error (RMSE), and mean absolute percentage error (MAPE). 

  - **Negative Control** -  Measurements were calculated between any two real *in vivo* profiles (excluding biological replicates), serving as a baseline for comparison. 
  - **AIVIVE** - Measurements were calculated between the optimized synthetic *in vivo* profiles generated by the model and its corresponding real profiles.
  - **Positive Control** - Measurements were calculated tween biological replicates within a treatment, providing a benchmark for real biological performance. 

- **Files**:
  - [`negative_control.py`](./applications/negative_control.py) - Python script for calculating the metrices for the negative control group as defined above.
  - [`cosine.py`](./applications/cosine.py) - For calculating the cosine similarity for AIVIVE group (generated profiles).
  - [`rmse.py`](./applications/rmse.py) - For calculating the rmse for AIVIVE group (generated profiles).
  - [`mape.py`](./applications/mape.py) - For calculating the mape for AIVIVE group (generated profiles).
  - [`positive_control.py`](./applications/positive_control.py) - For calculating the metrics for positive control group.
    
---

## **[Data Files](./data)**

This folder consists of the data files that were used in model development and training. 

  - [`final_rat_genes.csv`](./data/final_rat_genes.csv) - Rat S1500+ gene set used for AIVIVE model development and training.
  - [`open_tggates_cel_file_attribute.csv`](./data/open_tggates_cel_file_attribute.csv) - Open TG-GATEs metadata downloaded from [`https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html`]
  - [`rat_vitro_vivo_train_test`](https://zenodo.org/uploads/14984579?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY4Y2EwMmRhLWIzZDUtNGNjOS1hMjkwLTQ5ZDYwMzI2YzA4MCIsImRhdGEiOnt9LCJyYW5kb20iOiJiNGZjYTZmMGRmZDNhY2M4MGM0MzdlYzBhN2Q2NGQwOCJ9.ZuJ21SnVbRv5fywNeTpYbjlouhxtbGuF7Lm7U7qhkap8TLaXFY2-e2wL1hA_pMAUpl_3Be7X1c8km2UiUL6c1g) - Rat liver *in vitro* and *in vivo*(single dose) transcriptomic profiles that are RMA normalized. The data is split into train (80%) and test (20%) set based on unique compounds.
  - [`modules_genes`](./data/modules_genes) - Module gene sets used in local optimization
  - [`aop_overlap_genes.csv`](./data/aop_overlap_genes.csv) - AOP genes overlapped with rat S1500+ genes (used for AOP-Gene Expression Analysis)
  - [`necrosis_df.csv`](./data/necrosis_df.csv) - Pathological findings (from Open TG-GATEs)

---


## Installation

### Prerequisites

Before using this repository, ensure you have the following installed:
- Python (version 3.11.7)
- Tensorflow-GPU (version 2.4.1)
- R (version 4.4.1)
- Bioconductor (version 3.19)
- rat2302.db (version 3.13.0 from Bioconductor version 3.19 release)
- Other packages specified in the code scripts

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

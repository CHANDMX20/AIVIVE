# GenLocGAN

**GenLocGAN** is a novel **Generative Adversarial Network (GAN)** framework that combines a modified version of **CycleGAN** with local optimizers. It uses **modules** or **gene clusters** to facilitate the demonstration of ***In Vitro-In Vivo*** **Extrapolation (IVIVE)**.


---

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
  - [Preprocessing](#preprocessing)
  - [GenLocGAN Model Development & Training](#genlocgan-model-development--training)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

This repository contains code files and models for training the GenLocGAN framework and generating predictions for *in vitro-in vivo* extrapolation. Additionally, it provides scripts for applying the model to biologically relevant tasks, such as analyzing differentially expressed genes (DEGs), predicting necrosis, and other related applications.  

---

## Files

### **Preprocessing**

This folder contains the code and scripts for data preprocessing, specifically focusing on **RMA (Robust Multi-array Average) normalization** and **annotation** of gene expression data. These preprocessing steps are critical for preparing data before training the **GenLocGAN** model.

- **Key Features**:
  - **RMA Normalization**: Scripts to normalize gene expression data using the RMA method, which is widely used for processing microarray data.
  - **Gene Annotation**: Scripts to annotate gene expression data with relevant gene identifiers, pathways, and other biological annotations.

**Files**:
- [`rma_vivo_single.R`](./preprocessing/rma_vivo_single.R) - R script that performs **RMA normalization** on the raw gene expression data.
- [`rat_annotation.R`](./preprocessing/rat_annotation.R) - R script processes rat gene expression data by extracting probe IDs and merging them with gene annotations to generate a final dataset of unique rat genes used for GenLocGAN model development.

These preprocessing steps ensure that the gene expression data is properly prepared for the downstream tasks of training the **GenLocGAN** model.

---

### 2. **[GenLocGAN Model Development & Training](./training)**

This folder contains the core code for developing and training the **GenLocGAN** framework. It includes the implementation of the **modified CycleGAN** model, local optimizers, and other components essential for training the model.

- **Key Features**:
  - **Training scripts** for the **GenLocGAN** framework.
  - **CycleGAN model** implementation, including modifications specific to the GenLocGAN framework.
  - **Local optimizers** for enhancing the model's performance.
  - Code for handling **data preprocessing**, **data augmentation**, and **training loops**.

**Files**:
- `train_genlocgan.py` - Main script used to train the **GenLocGAN** model on your dataset.
- `cycle_gan.py` - Modified CycleGAN architecture tailored for GenLocGAN's framework.
- `optimizer.py` - Defines the custom optimizers used during training.
- `data_loader.py` - Contains data loading logic, including handling of in vitro and in vivo datasets.
- `config.json` - Configuration file containing model parameters, training options, and hyperparameters.

These scripts and configurations enable the training of the **GenLocGAN** model for generating predictions related to **in vitro-in vivo** extrapolation.

---

### 2. **[File Name](./path-to-file)**

Description of the next file, following the same format as above.

- **Key Features**:
  - Any important functions or capabilities.
  - Summary of what this file contributes to the project.

---

### 3. **[File Name](./path-to-file)**

Description of another file in the project, with a similar format.

- **Key Features**:
  - Mention any useful functions or methods.
  - Example results, if applicable.

---

## Installation

### Prerequisites

Before using this repository, ensure you have the following installed:
- List any required software or dependencies, e.g., Python, R, specific packages.
  - For Python:
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `scikit-learn`
  - For R:
    - `SPIEC-EASI`
    - `igraph`
    - `ggplot2`

### Installing Dependencies

For Python:
```bash
pip install -r requirements.txt

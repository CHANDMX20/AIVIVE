# GenLocGAN

**GenLocGAN** is a novel **Generative Adversarial Network (GAN)** framework that combines a modified version of **CycleGAN** with local optimizers. It uses **modules** or **gene clusters** to facilitate the demonstration of ***In Vitro-In Vivo*** **Extrapolation (IVIVE)**.


---

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
  - [Preprocessing](#preprocessing)
  - [GenLocGAN Model Development, Training & Predictions](#genlocgan-model-development-training--predictions)
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

**Files**:
- [`rma_vivo_single.R`](./preprocessing/rma_vivo_single.R) - R script that performs **RMA normalization** on the raw gene expression data which is widely used for processing microarray data
- [`rat_annotation.R`](./preprocessing/rat_annotation.R) - R script processes rat gene expression data by extracting probe IDs and merging them with gene annotations (relevant gene identifiers) to generate a final dataset of unique rat genes used for GenLocGAN model development.

These preprocessing steps ensure that the gene expression data is properly prepared for the downstream tasks of training the **GenLocGAN** model.

---

### **[GenLocGAN Model Development, Training & Predictions](./training)**

This folder contains the core code for developing and training the **GenLocGAN** framework. It includes the implementation of the **modified CycleGAN** model and local optimizer networks.

  - **Modified CycleGAN** framework for translating *in-vitro* transcriptomic profiles to *in-vivo* transcriptomic profiles.
  - **Local optimizers** for enhancing the model's performance.
  - **Predictions** for generating the test set predictions.

**Files**:
- [`vitro_vivo_GAN.py`](./training/vitro_vivo_GAN.py) - Modified CycleGAN script to train the **GenLocGAN** model on the IVIVE dataset.
- [`train_test_samples.py`](./training/train_test_samples.py) - Generating test set predictions using the optimal modified CycleGAN generator.
- [`optim_neural_net_#.py`](./training/modules) - Local optimizer neural network frameworks for specific modules, where `#` refers to the module number (e.g., `optim_neural_net_20.py`, `optim_neural_net_23.py`, etc.). These scripts contain implementations for training different modules.
- [`module_test_evals.py`](./training/modules/module_test_evals.py) - Generating test set predicitons for specific modules using the optimal local optimizers.

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

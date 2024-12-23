# Hierarchical-Spatial-Sequential-Modeling-of-Protein

This repository is the implementation of our Paper under review.

# Abstract
Protein-protein interactions play a fundamental role in biological systems. Accurate detection of protein-protein interaction sites (PPIS) remains a challenge. And, the methods of PPIS prediction based on biological experiments are expensive. Recently, a lot of computation-based methods have been developed and made great progress. However, current computational methods only focus on one form of protein, using only protein spatial conformation or primary sequence, and ignore the protein’s natural hierarchical structure. Here, we propose a novel network architecture, **HSSPPISP**, through **H**ierarchical and **S**patial-**S**equential modeling of protein for **P**rotein-**P**rotein **I**nteraction Sites Prediction. In this network, we represent protein as a hierarchical graph, in which a node in the protein is a residue (residue-level graph) and a node in the residue is an atom (atom-level graph). Moreover, we design a spatial-sequential block for capturing complex interaction relationships from spatial and sequential forms of protein. We evaluate HSSPPISP on public benchmark datasets and the predicting results outperform the comparative models. This indicates the effectiveness of hierarchical protein modeling and also illustrates that HSSPPISP has a strong feature extraction ability by considering spatial and sequential information at the same time.


## 1. Datasets and trainded models
The datasets used for training HSSPPISP and the trained models mentioned in our manuscrpit can be downloaded from https://pan.baidu.com/s/1H6xk9_mPtt8pSsSOV9pXEQ  （Password: PPIS）

## 2. Requirement
We implemented our method using PyTorch and Deep torch-geometric (PyG). Please install these tools for successfully running our code. Necessary installation instructions are available at the following links: 
* [python = 3.9.10](https://www.python.org/downloads/)
* [pytorch = 1.10.2](https://pytorch.org/get-started/locally/#start-locally)
* [torch-geometric = 2.4.0](https://pypi.org/project/torch-geometric/)

## 3. Usage
train.py provides the code to retrain the HSSPPI (hyperparameters can be reset in configs.py).

ProtT5 embeddings can be generated using bio_embeddings (https://github.com/sacdallago/bio_embeddings).

## 4. Citation
This repository is the implementation of our Paper under review.

## 5. References
[1] Min Zeng, Fuhao Zhang, Fang-Xiang Wu, Yaohang Li, Jianxin Wang, Min Li*. Protein-protein interaction site prediction through combining local and global features with deep neural networks[J]. Bioinformatics, 36(4), 2020, 1114–1120. DOI:10.1093/bioinformaticsz699.  
[2] Quadrini M, Daberdaku S, Ferrari C. Hierarchical representation for PPI sites prediction[J]. BMC Bioinformatics, 2022, 23(1):1-34. DOI:10.1186/s12859-022-04624-y.  

## 6. Contact
For questions and comments, feel free to contact: ieslu@zzu.edu.cn.

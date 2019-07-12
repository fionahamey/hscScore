# hscScore

## Author: Fiona Hamey
## Summary: GitHub repository for Hamey & Gottgens, in preparation

## Introduction
Haematopoietic stem cells (HSCs) represent a rare cell type in mouse bone marrow, but these cells play an important role in maintaining the blood system. As identifying this population in single-cell gene expression profiling experiments remains a challenge, we developed the method hscScore score for location HSCs in single-cell RNA-sequencing (scRNA-seq) data. This repository contains the code using for training the hscScore model and reproducing the analysis described in the paper. We also provide an example of how to apply the model to your own data.

## Running hscScore on your own data
To run hscScore on your own data you need to provide a gene count matrix for mouse scRNA-seq data with dimensions cells x genes. Columns should be labelled with common gene names. The [hscScore_demonstration_notebook.ipynb notebook](https://nbviewer.jupyter.org/github/fionahamey/hscScore/blob/master/analysis_notebook_hamey_and_gottgens.ipynb) demonstrates how you can apply the hscScore model. The trained model can be downloaded as a python pickle object from [Zenodo](https://doi.org/10.5281/zenodo.3332150).

## Reproducing analysis from the paper
The file hsc_score_parameter_search.py shows how GridSearchCV was used to identify the best scoring parameter combinations for the different models tested for predicting the HSC-score on scRNA-seq data. 

The notebook analysis_notebook_hamey_and_gottgens.ipynb shows how the final hscScore model was trained based on the results of this parameter search, and can be used to reproduce the plots shown in the paper. Supporting data for this notebook can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.3303783).

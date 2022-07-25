# Explainable ERP Fraud Detection
This repository contains code for the paper 'Towards Explainable Occupational Fraud Detection' 
published at the 7th Workshop on MIning DAta for financial applicationS (MIDAS) as part of the 
European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD) 2022.

## Abstract
Occupational fraud within companies currently causes losses of around 5% of company revenue each year. 
While enterprise resource planning systems can enable automated detection of occupational fraud 
through recording large amounts of company data, the use of state-of-the-art machine learning approaches 
in this domain is limited by their untraceable decision process. 
In this study, we evaluate whether machine learning combined with explainable artificial intelligence 
can provide both strong performance and decision traceability in occupational fraud detection. 
We construct an evaluation setting that assesses the comprehensibility of machine learning-based 
occupational fraud detection approaches, and evaluate both performance and comprehensibility 
of multiple approaches with explainable artificial intelligence. 
Our study finds that high detection performance does not necessarily indicate good explanation quality, 
but specific approaches provide both satisfactory performance and decision traceability, 
underlining the suitability of machine learning for practical application in occupational fraud detection 
and the importance of research evaluating both performance and comprehensibility together.

## Contained Materials
The `data` folder contains the ERP fraud detection data of Tritscher et al. [[1]](https://arxiv.org/abs/2206.04460).
The full data is available at https://professor-x.de/erp-fraud-data.

Results of the hyperparameter studies from both paper experiments can be found unter `outputs/summary`.

Additionally, the folder `output/explanation` contains the generated SHAP explanations from both experiments.

## Usage
Hyperparameter studies and training for fraud detection approaches can be conducted through `param_search.py`.

Stored models can be explained with SHAP [[2]](https://github.com/slundberg/shap) using `run_xai.py`.

Interactive visualization of explanations is possible through `visualize_xai.ipynb`.

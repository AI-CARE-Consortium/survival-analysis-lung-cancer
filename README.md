# Survival Analysis for Lung Cancer Patients: A Comparison of Cox Regression and Machine Learning Models 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10377082.svg)](https://doi.org/10.5281/zenodo.10377082)


### Authors: Sebastian Germer, Christiane Rudolph, Louisa Labohm, Alexander Katalinic, Natalie Rath, Katharina Rausch, Heinz Handels and the AI-CARE consortium

# How to Use
- Install required packages (Pytorch, Scikit-Surv, Scikit-Learn, Tabnet, SHAP) via conda (see enviroment.yaml)
- Edit the ```base_dir``` and the ```device``` for NN training ("cpu", "cuda:1",...) in your config.yaml
- Run ```python parameterized_training_kfolds.py``` with the following arguments:
  - ```--imputation_method```: Which imputation method to use ("none", "KNNImputer", "SimpleImputer", "MissForest")
  - ```--model```: Which model to use ("rsf", "cox", "deep_surv", "tabnet")
  - ```--deep_surv_model```: If model is deep_surv, which specific deep_surv model to use ("minimalistic_network")
  - ```--tnm```: Use the TNM Subset instead of the UICC subset (optional)
  - ```--one-hot```: Use one-hot encoding instead of label encoding (optional)
  - ```--loss ```: Which loss function to use ("pll", "mse")

- Now, your chosen model is fitted and evaluated over 50 hyperparameter search epochs

- For visualization, run ```python parameterized_evaluation_kfolds.py``` afterwards with the same parameters as above

# How to Cite
A journal paper in currently in writing. In the meantime, you can cite this repository using
> Germer, S., Rudolph, C., Labohm, L., Katalinic, A., Rath, N., Rausch, K., & Handels, H. (2023). Survival Analysis for Lung Cancer Patients: A Comparison of Cox Regression and Machine Learning Models. Zenodo. https://doi.org/10.5281/zenodo.10377082


 # A2A Binary Classifier

This repository contains the code for a binary classification model to predict the activity of compounds with respect to the adenosine A2a receptor. We tried different models, including logistic regression, decision tree, random forest, and support vector machine.

## Prerequisites
**Python 3.7**

Most of these are okay to install with **pip**. Install the necessary libraries for this project using this command to your environment

`pip install -r requirements.txt`

## Data

The data used to train and test the models is from the DUD-E database. Raw dataset contains 3050 active compounds, 192 inactive compounds and 31150 decoys. We divided our data in two classes (active and inactive) by adding decoys to inactive compounds. We obtained a balanced dataset with 6100 data points containing two classes.

## Feature generation

After resampling the dataset we used open-source Python library RDKit for feature generation from molecular structure.

`data_processing.ipynb`

## Feature selection
We also performed feature selection using:
- Recursive Feature Elimination (RFE)
- Feature selection using Random Forest Classifier - Optimal number of features : 25
- Permutation importance for SVC and Logistic

`feature_selection.ipynb`

## Models

We trained several different models on the data, including:

* Logistic regression

  `log_regr.ipynb`
  
* Decision tree
* Random forest

  `dt_and_rf.ipynb`
  
* Support vector machine

  `svm.ipynb`

The models were evaluated using the area under the curve (AUC) of the receiver operating characteristic (ROC) curve. The best model was the random forest model, which achieved an AUC of 0.978.

## Results

The results of the models are shown in the table below.

| Model | Accuaracy | AUC |
|---|---|---|
| Logistic regression | 0.945 | 0.947 |
| Decision tree | 0.948 | 0.948 |
| Random forest | 0.978 | 0.978 |
| Support vector machine | 0.975 | 0.975 |

## Conclusion

The random forest model was the best model for predicting the activity of compounds with respect to the adenosine A2a receptor. This model could be used to identify new compounds that are active at this receptor.

## Reference

     	1. Chen, L. et al. Hidden bias in the DUD-E dataset leads to misleading performance of deep learning in structure-based virtual screening. PLOS ONE 14, e0220113 (2019).

## License

The software is licensed under the MIT license (see LICENSE file), and is free and provided as-is.

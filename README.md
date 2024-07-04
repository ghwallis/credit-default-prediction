# Credit Card Default Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![SAS](https://img.shields.io/badge/SAS-EM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
   - [SAS EM](#sas-em)
   - [Python](#python)
3. [Model Selection and Hyperparameter Optimization](#model-selection-and-hyperparameter-optimization)
   - [Logistic Regression](#logistic-regression)
   - [Decision Tree](#decision-tree)
   - [Random Forest](#random-forest)
   - [Neural Network](#neural-network)
4. [Cross-Validation and Model Comparison](#cross-validation-and-model-comparison)
5. [Results and Conclusion](#results-and-conclusion)
6. [Dependencies and Installation](#dependencies-and-installation)
7. [Appendices](#appendices)

## Introduction

This project aims to predict whether a credit card customer will default using logistic regression, decision tree, neural network, and random forest models. The prediction is based on their latest 6-month payment record and other information. We utilized both SAS EM and Python for our analysis. Data preprocessing was conducted first, followed by hyperparameter optimization for the models. The final model was selected based on 70/30 cross-validation results, comparing metrics such as misclassification rate, recall, specificity, precision, accuracy, and F1 score.

This project was submitted as part of a Machine Learning class during my Master's program.

## Data Preprocessing

### SAS EM

In SAS EM, data preprocessing involved replacing outliers and imputing missing values. Outliers were replaced with missing values and then imputed using tree methods for both interval and nominal/binary attributes. No variables were dropped as none had more than 50% missing values.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/72739756-d956-4f90-8057-40fe2c2ff10a" alt="SAS1" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/0ce1fec2-ec5c-487a-bbe9-52db28f0e5b6" alt="SAS2" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/74e45547-8815-483c-bb8f-23c224b192e3" alt="SAS3" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/562ca834-be9b-4657-87b3-426c4055cafb" alt="SAS4" width="45%">
</div>

### Python

In Python, data preprocessing was done using the `Class_replace_impute_encode` class with different configurations for different models. An attribute map was created, and preprocessing was tailored for each model accordingly.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/92180a04-fc63-443f-93cb-be4b57cb8c1b" alt="Python1" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/0dabca83-8a18-4c1e-8378-4b3e6c757b8f" alt="Python2" width="45%">
</div>

## Model Selection and Hyperparameter Optimization

### Logistic Regression

#### SAS EM

Six different configurations were tested using SAS EM. The HP Stepwise method showed the best results with the lowest misclassification rate and highest recall, specificity, precision, accuracy, and F1 score.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/4944d329-4178-409c-b478-767af8579e82" alt="SAS5" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/62809ebb-f403-4647-9c1f-df53c173eed4" alt="SAS6" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/3f3eeb54-7f8a-4a30-9fa7-4be93a9bb0b1" alt="SAS7" width="45%">
</div>

#### Python

A self-defined stepwise function was used for hyperparameter optimization due to the lack of built-in functions in Python. The function performed 10-fold cross-validation to select the best model configuration.

### Decision Tree

#### SAS EM

Ten different configurations were tested, with the Non-HP Decision Tree with a depth of 12 performing best.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/d04a652e-ddd6-43d4-bbac-693c42a8f246" alt="SAS8" width="45%">
</div>

#### Python

Various depths (5, 6, 7, 8, 10, 12) were tested using the Gini index. The Decision Tree with depth 6 performed best in terms of accuracy, precision, and F1 score.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/0a74f667-5d9d-4caa-88f3-296967314d0f" alt="Python3" width="45%">
</div>

### Random Forest

#### SAS EM

The Random Forest model with P=1 showed the best performance.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/df07e7ef-83c9-4f32-8b58-7e6e41fde882" alt="SAS10" width="45%">
</div>

#### Python

Random Forest models were tested with different numbers of trees and maximum features. The model with 15 trees and 0.7 maximum features performed best.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/d3a31bab-801e-4386-8f32-6b7e545a014f" alt="Python4" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/ec0b9830-5fd5-429b-a3e1-b1b91ea52a80" alt="Python5" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/97e0b7d6-e112-4daf-97b9-db00a44baa33" alt="Python6" width="45%">
</div>

### Neural Network

#### SAS EM

Ten different configurations were tested. The HP model with 54 perceptrons performed best.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/431e6c2f-59bd-49b8-8eb4-6ad851eabafe" alt="SAS11" width="45%">
</div>

#### Python

Various configurations were tested, with the Neural Network having (4,3) perceptrons showing the best performance.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/0b80cc37-06ee-49fa-950f-821bc712d9da" alt="Python7" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/745e88fa-02e7-441e-92fd-fa504fd85056" alt="Python8" width="45%">
</div>

## Cross-Validation and Model Comparison

### SAS EM

The final 70/30 cross-validation determined the Non-HP Decision Tree with depth 12 as the best model. However, all four models did not produce desirable results as their recalls were lower or close to 50%. This means that the models were not very helpful in predicting true positives, i.e., whether a customer will default on his/her account. Logistic regression was especially disappointing as its F1 score was also lower than 40%.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/21806124-229c-400f-ab4c-d639446bc368" alt="SAS12" width="45%">
</div>

### Python

The final 70/30 cross-validation determined the Random Forest model as the best. However, SAS EM's Non-HP Decision Tree with depth 12 outperformed the Python models.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/390c4f80-4573-4432-949f-08937308614a" alt="Python9" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/f282425e-cd5f-4efe-959f-f2bc8674351e" alt="Python10" width="46%">
</div>

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/99402670-1590-4734-8292-92c0834ee12a" alt="Python11" width="45%">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/e6f4056d-f960-4cc7-85f4-bc0e78a00426" alt="Python12" width="42%">
</div>

<img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/46bea43f-0d05-4b71-86fd-a81e391a4030" alt="Python13">


## Results and Conclusion

Based on our analysis, the Non-HP Decision Tree model from SAS EM was found to be the most effective in predicting credit card defaults. While the models in Python also performed well, the SAS EM model provided slightly better results.

<div style="display: flex; flex-wrap: wrap;">
  <img src="https://github.com/ghwallis/credit-default-prediction/assets/36977382/bbc68d71-b240-4410-898e-82576db7a7d4" alt="Python14" width="65%">
</div>

## Dependencies and Installation

Ensure you have the following dependencies installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- SAS EM (for SAS analysis). **Note: User will need a SAS license**

Install the Python dependencies using:

```bash
pip install pandas numpy scikit-learn
```

### Appendices

#### Python Code
The Python scripts used for data preprocessing, model training, and evaluation are included in the in the **_credit_default_model.py_** file in this directory .


#### Data
The dataset used for this project can be found in the data directory in the **_CreditCard_Defaults.xlsx_** file.

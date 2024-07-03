# -*- coding: utf-8 -*-
# @Time    : 3/4/2018 11:46
# @Author  : sfirdaws

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import ast

# Import custom classes
from Class_replace_impute_encode import ReplaceImputeEncode

# Define subset selection function for logistic regression
def subset_selection(data, target):
    accuracy = {}
    Y = np.asarray(data[target])
    X = data.drop(target, axis=1)
    lgr = LogisticRegression()
    variables = X.columns.values
    without_last = variables
    last = []

    for p in range(1, len(variables)):
        current_combinations = {}
        for var in without_last:
            new_col = last + [var]
            d_test = pd.DataFrame(X, columns=new_col)
            X_1 = np.asarray(d_test)
            lgr_score = cross_val_score(lgr, X_1, Y, cv=10).mean()
            current_combinations[str(new_col)] = lgr_score

        best_combination = max(current_combinations.items(), key=lambda x: x[1])
        best_vars = ast.literal_eval(best_combination[0])
        last = best_vars
        without_last = list(set(variables) - set(last))
        accuracy[best_combination[0]] = best_combination[1]

    best_subset = max(accuracy.items(), key=lambda x: x[1])
    print("Best subset of variables:", best_subset[0])
    return best_subset

# Load and preprocess data
df = pd.read_excel('CreditCard_Defaults.xlsx')
n_obs = df.shape[0]
initial_missing = df.isnull().sum()

# Check for attributes with more than 50% missing values and drop them
for feature in df.columns:
    if initial_missing[feature] > (n_obs / 2):
        print(f"{feature} has {initial_missing[feature]} missing values. Dropping this attribute.")
        df = df.drop(feature, axis=1)

# Drop 'Customer' attribute
df = df.drop('Customer', axis=1)

# Define attribute map for ReplaceImputeEncode
attribute_map = {
    'Default': [1, (1, 0), [0, 0]],
    'Gender': [1, (1, 2), [0, 0]],
    'Education': [2, (0, 1, 2, 3, 4, 5, 6), [0, 0]],
    'Marital_Status': [2, (0, 1, 2, 3), [0, 0]],
    'card_class': [2, (1, 2, 3), [0, 0]],
    'Age': [0, (20, 80), [0, 0]],
    'Credit_Limit': [0, (100, 80000), [0, 0]],
    'Jun_Status': [0, (-2, 8), [0, 0]],
    'May_Status': [0, (-2, 8), [0, 0]],
    'Apr_Status': [0, (-2, 8), [0, 0]],
    'Mar_Status': [0, (-2, 8), [0, 0]],
    'Feb_Status': [0, (-2, 8), [0, 0]],
    'Jan_Status': [0, (-2, 8), [0, 0]],
    'Jun_Bill': [0, (-12000, 32000), [0, 0]],
    'May_Bill': [0, (-12000, 32000), [0, 0]],
    'Apr_Bill': [0, (-12000, 32000), [0, 0]],
    'Mar_Bill': [0, (-12000, 32000), [0, 0]],
    'Feb_Bill': [0, (-12000, 32000), [0, 0]],
    'Jan_Bill': [0, (-12000, 32000), [0, 0]],
    'Jun_Payment': [0, (0, 60000), [0, 0]],
    'May_Payment': [0, (0, 60000), [0, 0]],
    'Apr_Payment': [0, (0, 60000), [0, 0]],
    'Mar_Payment': [0, (0, 60000), [0, 0]],
    'Feb_Payment': [0, (0, 60000), [0, 0]],
    'Jan_Payment': [0, (0, 60000), [0, 0]],
    'Jun_PayPercent': [0, (0, 1), [0, 0]],
    'May_PayPercent': [0, (0, 1), [0, 0]],
    'Apr_PayPercent': [0, (0, 1), [0, 0]],
    'Mar_PayPercent': [0, (0, 1), [0, 0]],
    'Feb_PayPercent': [0, (0, 1), [0, 0]],
    'Jan_PayPercent': [0, (0, 1), [0, 0]],
}

# Encode and impute data for different models
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', interval_scale=None, drop=True, display=True)
rie2 = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', interval_scale=None, drop=False, display=True)
rie3 = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', interval_scale='std', drop=False, display=True)

df_lgr = rie.fit_transform(df)  # For logistic regression
df_tree = rie2.fit_transform(df)  # For decision tree and random forest
df_NN = rie3.fit_transform(df)  # For neural network

# Define target variable
y = df_lgr['Default']

# Logistic Regression with subset selection
X_lgr = df_lgr.drop('Default', axis=1)
selected_columns = subset_selection(df_lgr, 'Default')
selected_features = selected_columns[0].replace('\'', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
X_lgr_selected = X_lgr[selected_features]

# Evaluate Logistic Regression model
lgr = LogisticRegression().fit(X_lgr_selected, y)
score_list = ['accuracy', 'recall', 'precision', 'f1']
scores = cross_validate(lgr, X_lgr_selected, y, scoring=score_list, return_train_score=False, cv=10)
print("\nLogistic Regression Results:")
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
for metric in score_list:
    print(f"{metric:.<13s}{scores['test_' + metric].mean():>7.4f}{scores['test_' + metric].std():>10.4f}")

# Decision Tree model evaluation
X_tree = df_tree.drop('Default', axis=1)
depth_list = [5, 6, 7, 8, 10, 12, 15, 20, 25]
print("\nDecision Tree Results:")
for depth in depth_list:
    dtc = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, min_samples_split=5).fit(X_tree, y)
    scores = cross_validate(dtc, X_tree, y, scoring=score_list, return_train_score=False, cv=10)
    print(f"\nMaximum Tree Depth: {depth}")
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for metric in score_list:
        print(f"{metric:.<13s}{scores['test_' + metric].mean():>7.4f}{scores['test_' + metric].std():>10.4f}")

# Random Forest model evaluation
estimators_list = [10, 15, 20, 25, 30, 35]
print("\nRandom Forest Results:")
for estimators in estimators_list:
    clf = RandomForestClassifier(n_estimators=estimators, max_depth=8, random_state=60616, max_features=4).fit(X_tree, y)
    scores = cross_validate(clf, X_tree, y, scoring=score_list, return_train_score=False, cv=10)
    print(f"\nNumber of Estimators: {estimators}")
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for metric in score_list:
        print(f"{metric:.<13s}{scores['test_' + metric].mean():>7.4f}{scores['test_' + metric].std():>10.4f}")

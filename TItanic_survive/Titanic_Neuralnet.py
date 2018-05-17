# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from sklearn import grid_search
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv as csv

import time #時間計測のための拡張モジュール

plt.style.use('ggplot')
start = time.time()

df = pd.read_csv("/Users/Masataka/Documents/GitHub/Kaggle/TItanic_survive/train.csv")

# training rabel
Y_train = df.copy()
#Y_train['Survived'] = Y_train['Survived'].map({"yes":1, "no":0})
Y_train = Y_train[Y_train['Survived'].notnull()]
Y_train = Y_train.iloc[:, 1].values #  income のみ

X_train = df.iloc[:, 2:12]#  apart y 

#落とす
colnames_drop = ['Name','Ticket','Cabin']
X_train = X_train.drop(colnames_drop, axis=1)
'''
#欠損
colnames_notnull = ['Age','Embarked']
X_train = X_train[X_train[colnames_notnull].notnull()]
'''

# create dummy variables of categoly variables
colnames_categorical = ['Sex','Embarked']
X_dummy = pd.get_dummies(X_train[colnames_categorical], drop_first=True)

# conbination of dummies
X_train = pd.merge(X_train, X_dummy, left_index=True, right_index=True)

# 使わない、重複している列の削除
X_train = X_train.drop(colnames_categorical, axis=1)

# Complement the missing values of "Age" column with average of "Age"
median_age = X_train["Age"].dropna().median()
if len(X_train.Age[X_train.Age.isnull()]) > 0:
  X_train.loc[(X_train.Age.isnull()), "Age"] = median_age


ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)

#Neural Net
clf = MLPClassifier(solver="adam",max_iter=30,hidden_layer_sizes=(10,10))
clf.fit(X_train, Y_train)


'''
# 分類し、誤り率を出
# 5分割のStratifiedKFoldでそれぞれスコアを算出

scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=10,)
print ("scores: ", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
# 学習モデルの評価


pdd = pd.read_csv("/Users/Masataka/Documents/GitHub/Kaggle/TItanic_survive/test.csv")

ids = pdd["PassengerId"].values

# training rabel
X_test = pdd.copy()


X_test = pdd.iloc[:, 1:12]#  apart y 

#落とす
colnames_drop2 = ['Name','Ticket','Cabin']
X_test = X_test.drop(colnames_drop2, axis=1)

# create dummy variables of categoly variables
colnames_categorical2 = ['Sex','Embarked']
X_dummy1 = pd.get_dummies(X_test[colnames_categorical2], drop_first=True)

# conbination of dummies
X_test= pd.merge(X_test, X_dummy1, left_index=True, right_index=True)

# 使わない、重複している列の削除
X_test = X_test.drop(colnames_categorical2, axis=1)

# Complement the missing values of "Age" column with average of "Age"
median_age = X_test["Age"].dropna().median()
if len(X_test.Age[X_test.Age.isnull()]) > 0:
  X_test.loc[(X_test.Age.isnull()), "Age"] = median_age

median_fare = X_test["Fare"].dropna().median()
if len(X_test.Fare[X_test.Age.isnull()]) > 0:
  X_test.loc[(X_test.Fare.isnull()), "Fare"] = median_fare
  
  
ms = MinMaxScaler()
X_test = ms.fit_transform(X_test)

pred = clf.predict(X_test)



elapsed_time = time.time() - start
print(elapsed_time)


# export result to be "titanic_submit.csv"
submit_file = open("titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, pred))
submit_file.close()
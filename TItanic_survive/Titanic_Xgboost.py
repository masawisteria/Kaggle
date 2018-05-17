#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:49:21 2017

@author: Masataka
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb
import csv as csv
from sklearn.preprocessing import MinMaxScaler


plt.style.use('ggplot')


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




#2. xgboostモデルの作成
clf = xgb.XGBClassifier()

#2.1 ハイパーパラメータ探索
clf_cv = GridSearchCV(clf, {'max_depth': [5,6,7], 'n_estimators': [50,100,200]}, verbose=1)
clf_cv.fit(X_train, Y_train)
print (clf_cv.best_params_, clf_cv.best_score_)

#2.2 改めて最適パラメータで学習
clf = xgb.XGBClassifier(**clf_cv.best_params_)
clf = clf.fit(X_train, Y_train)

#2.3 学習モデルの保存、読み込み
import pickle
pickle.dump(clf, open("model.pkl", "wb"))
clf = pickle.load(open("model.pkl", "rb"))


# 分類し、誤り率を算出
# 5分割のStratifiedKFoldでそれぞれスコアを算出

scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=10,)
print ("scores: ", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

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



pred = clf.predict(X_test)
'''
report = classification_report(Y_train, pred)
accuracy = accuracy_score(Y_train, pred)


print (report)
print (accuracy)

probas_ = clf.predict_proba(X_train)
fpr, tpr, thresholds= roc_curve(Y_train, probas_[:,1])
precision, recall, thresholds = precision_recall_curve(Y_train, probas_[:,1])
area = auc(recall, precision)
print ("Area Under Curve: {0:.3f}".format(area))

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr)
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


np.savetxt('/Volumes/MASATAKA/Statistic/TeamProjectdocdata17/xg.csv',probas_,delimiter=",")  
'''
# export result to be "titanic_submit.csv"
submit_file = open("titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids, pred))
submit_file.close()
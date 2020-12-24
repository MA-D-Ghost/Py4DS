
'''
Ho ten: Ton Thien Minh Anh
MSSV: 18110049
Bai thuc hanh Lab08
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# LOAD DATA
path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
df = pd.read_csv(path, header = None)

print(df.head())
print('Data shape = ',df.shape)
print(" ========== DATA INFO ========== ")
print(df.info())



# Separate between feature (X) and label (Y)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=20)

# Cross-validation
from sklearn.model_selection import cross_val_score
from sklearn import svm

print(' ========== CROSS - VALIDATION ========== ')
print("\n")


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
print(scores)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# K - FOLDS
from sklearn.model_selection import KFold
print("\n")
print(' ========== K - FOLDS ========== ')


kf = KFold(n_splits=5)
scores = cross_val_score(clf, X_train, y_train, cv=kf)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# LEAVE ONE OUT 

from sklearn.model_selection import LeaveOneOut
print("\n")
print(' ========== LEAVE ONE OUT ========== ')

loo = LeaveOneOut()
scores = cross_val_score(clf, X_train, y_train, cv=loo)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# STRATIFIED KFOLD
from sklearn.model_selection import StratifiedKFold

print("\n")
print(' ========== STRATIFIED KFOLD ========== ')

skf = StratifiedKFold(n_splits=3)

scores = cross_val_score(clf, X_train, y_train, cv=skf)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



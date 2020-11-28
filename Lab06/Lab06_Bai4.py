'''
Ho ten: Ton Thien Minh Anh
MSSV: 18110049
Bai thuc hanh Lab06 - Bai 4

Using Santander_train.csv, Santander_test.csv
Perform data dimensionality reduction and feature selection (at least 5 technologies)
Build model
'''

# Import thu vien
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import metrics to evaluate the performance of each model
import sklearn.metrics as metrics

# ================ LOAD DATA ================

# Load Data
path = '/home/ton-anh/Python Cho KHDL/Week6/Santander_train.csv'
train_df = pd.read_csv(path)

X=train_df.drop(columns=['TARGET'])
y=train_df['TARGET']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)

print("Shape cua X_train la: ",x_train.shape)
print("Shape cua X_test la: ",x_test.shape)


# =============== CONSTANT FEATURES + BUILD MODEL =============== 

print('========= CONSTANT FEATURES =========')
from sklearn.feature_selection import VarianceThreshold

constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(x_train)
len(x_train.columns[constant_filter.get_support()])

x_train_filter = constant_filter.transform(x_train)
x_test_filter = constant_filter.transform(x_test)

print("Shape cua X_train sau khi filter la: ",x_train_filter.shape)
print("Shape cua X_test sau khi filter la: ",x_test_filter.shape)

# Decision Tree
clf=DecisionTreeClassifier()
clf.fit(x_train_filter,y_train)
y_pred = clf.predict(x_test_filter)

print("DCT Accuracy: ",metrics.accuracy_score(y_test,y_pred))
# Accuracy ~ 0.928111
print('========= Tap Santander Test.csv =========')
path2 = '/home/ton-anh/Python Cho KHDL/Week6/Santander_test.csv'
test_df = pd.read_csv(path2)

print(test_df.shape)

test_df_filter = constant_filter.transform(test_df)
print("Shape cua test_df sau khi filter la: ",test_df_filter.shape)

# =============== QUASI CONSTANT FEATURES + BUILD MODEL =============== 
print('========= QUASI CONSTANT FEATURES =========')
qconstant_filter = VarianceThreshold(threshold=0.01)
qconstant_filter.fit(x_train)
len(x_train.columns[qconstant_filter.get_support()])

x_train_filter2 = qconstant_filter.transform(x_train)
x_test_filter2 = qconstant_filter.transform(x_test)

print("Shape cua X_train sau khi filter la: ",x_train_filter2.shape)
print("Shape cua X_test sau khi filter la: ",x_test_filter2.shape)

# Decision Tree
clf=DecisionTreeClassifier()
clf.fit(x_train_filter2,y_train)
y_pred = clf.predict(x_test_filter2)

print("DCT Accuracy: ",metrics.accuracy_score(y_test,y_pred))
# Accuracy ~ 0.929623

print('========= Tap Santander Test.csv =========')

print(test_df.shape)

test_df_filter2 = constant_filter.transform(test_df)
print("Shape cua test_df sau khi filter la: ",test_df_filter2.shape)
# =============== REMOVE DUPLICATED FEATURES + BUILD MODEL =============== 
print('========= REMOVE DUPLICATED FEATURES =========')

x_train_T = x_train.T
print(x_train_T.shape)

print(x_train_T.duplicated().sum())

unique_features = x_train_T.drop_duplicates(keep='first').T
print(unique_features.shape)
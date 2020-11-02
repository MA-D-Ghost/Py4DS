# Python For Data Science: Lab03
# Ho ten: Ton Thien Minh Anh
# MSSV: 18110049
# Bai 1

# Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Import ML Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import metrics to evaluate the performance of each model
import sklearn.metrics as metrics


# LOAD DATA
df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week3/creditcard.csv')

# EDA
print(df.info())

# Correlation Map
plt.figure(figsize=(32,32))
sns.heatmap(df.corr(),linewidth=.1,cmap="YlGnBu",annot=True)

# Drop duplicates
print(df.Class.value_counts())
rows = df.shape[0]
df.drop_duplicates(subset=df.columns.values[:-1],keep='first',inplace=True)
print(rows-df.shape[0],"duplicated Rows has been removed")

# Drop the rows where at least one elements is missing
df = df.dropna()

# Remove Outliers
plt.subplot(2,2,1)
ax = sns.boxplot(x=df["V1"])
plt.subplot(2,2,2)
ax = sns.boxplot(x=df["V2"])
plt.subplot(2,2,3)
ax = sns.boxplot(x=df["V3"])
plt.subplot(2,2,4)
ax = sns.boxplot(x=df["V4"])

# Nhận xét: Khi vẽ biểu đồ  Boxplot của một số cột giá trị từ V1 đến V4 thì mỗi feature đều có các Outliers
# => Cần xử lý các Outliers trước khi train Model

for i in range(29):
    Q1 =  df[df.columns[i]].quantile(0.25)
    Q3 = df[df.columns[i]].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df[df.columns[i]] >= Q1 - 1.5 * IQR) & (df[df.columns[i]] <= Q3 + 1.5 *IQR)
    df = df.loc[filter]

# SPLIT DATA
X = df.drop(['Class'],axis=1)
y = df['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

# TRAIN MODEL FIRST TIME

# Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Tree Prediction Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# SVM Classifier
clf=SVC()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("SVM Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train,y_train)
y_pred = rdf.predict(X_test)
print("RDF: ",metrics.accuracy_score(y_test,y_pred))
print("\n")


# NORMALIZE 

normalizer = preprocessing.Normalizer()
normalized_X_train = normalizer.fit_transform(X_train)

# TRAIN MODEL SECOND TIME

# Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(normalized_X_train,y_train)
y_pred = clf.predict(X_test)

print("Tree Prediction Accuracy 2: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# SVM Classifier
clf=SVC()
clf.fit(normalized_X_train,y_train)
y_pred = clf.predict(X_test)
print("SVM Accuracy 2: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(normalized_X_train,y_train)
y_pred = rdf.predict(X_test)
print("RDF Accuracy 2: ",metrics.accuracy_score(y_test,y_pred))
print("\n")



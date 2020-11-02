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
from sklearn.naive_bayes import GaussianNB

# Import metrics to evaluate the performance of each model
import sklearn.metrics as metrics


# LOAD DATA
df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week3/FIFA2018Statistics.csv')

# EDA
print(df.info())

# Find out how many nulls are found in each column in dataset
print(df.isnull().sum())

# Change type Man Of The Match
df['Man of the Match'] = df['Man of the Match'].map({'Yes': 1, 'No': 0})

# Describe
print(df.describe())

# Correlation Map
plt.figure(figsize=(32,32))
sns.heatmap(df.corr(),linewidth=.1,cmap="YlGnBu",annot=True)

# Nhận xét: 
# 1/ 'Man Of The Match' có mối tương quan chặt chẽ
#  với lại 'Goal Scored', 'On-Target', 'Corners', 'Attempts', 'free Kicks', 'Yellow Card', 'red', 'Fouls Committed', 'Own goal Time' 
# 2/ Còn những feature sau đây không có mối tương quan
#  với 'Man Of The Match': 'Blocked', 'OffSides', 'Saves','Distance Covered (Kms)', 'Yellow & Red', '1st Goal', 'Goals in PSO'
# => Ta sẽ drop bớt những feature không liên quan

# Drop columns
df.drop(["Blocked","Offsides","Saves","Distance Covered (Kms)","Yellow & Red","1st Goal","Goals in PSO","Own goals","Own goal Time","Round","Date","Team","Opponent","PSO"],axis=1,inplace=True)

print(df.info())

# Drop duplicates
rows = df.shape[0]
df.drop_duplicates(subset=df.columns.values[:-1],keep='first',inplace=True)
print(rows-df.shape[0],"duplicated Rows has been removed")

# Drop the rows where at least one elements is missing
df = df.dropna()

# Remove Outliers

# Boxplot
plt.subplot(2,2,1)
ax = sns.boxplot(x=df["Goal Scored"])
plt.subplot(2,2,2)
ax = sns.boxplot(x=df["Ball Possession %"])
plt.subplot(2,2,3)
ax = sns.boxplot(x=df["Attempts"])
plt.subplot(2,2,4)
ax = sns.boxplot(x=df["On-Target"])

# Nhận xét: Khi vẽ biểu đồ  Boxplot của một số cột giá trị đều có các Outliers
# => Cần xử lý các Outliers trước khi train Model

for i in range(12):
    Q1 =  df[df.columns[i]].quantile(0.25)
    Q3 = df[df.columns[i]].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df[df.columns[i]] >= Q1 - 1.5 * IQR) & (df[df.columns[i]] <= Q3 + 1.5 *IQR)
    df = df.loc[filter]

# SPLIT DATA
X = df.drop(['Man of the Match'],axis=1)
y = df['Man of the Match']

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

# Logistics Regression Classifier
lr = LogisticRegression()
lr = lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print("LR: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

#Naive Bayes
nb =  GaussianNB()
nb.fit(X_train, y_train)
y_pred=nb.predict(X_test)

print("Accuracy of naive bayees algorithm: ",nb.score(X_test,y_test))

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

# Logistics Regression Classifier
lr = LogisticRegression()
lr = lr.fit(normalized_X_train,y_train)
y_pred = lr.predict(X_test)

print("LR 2: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

#Naive Bayes
nb =  GaussianNB()
nb.fit(normalized_X_train, y_train)
y_pred=nb.predict(X_test)

print("Accuracy of naive bayees algorithm 2: ",nb.score(X_test,y_test))

# Nhận xét: Accuracy sau khi Normalize Data thấp hơn Accuracy trước khi Normalize. Như vậy, đối với Data này, việc Normalize là không hiệu quả và không nên thực hiện

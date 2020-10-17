# Python for Data Science: Week 1
# MSSV: 18110049
# Ho ten: Ton Thien Minh Anh 

# Load Libraries
import pandas as pd 
import numpy as np 
import scipy as sp 
from sklearn.model_selection import train_test_split

# Import ML Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import metrics to evaluate the performance of each model
import sklearn.metrics as metrics

# Load Dataframe
df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week1/mushrooms.csv')
print(df.head())

y = df['class']
X=df.drop('class',axis = 1)
print(y.head())

# Convert data to integer or float using Label Encoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for column in df.columns: 
    df[column]=labelencoder.fit_transform(df[column])

y = df['class']
X=df.drop('class',axis = 1)

print(df.dtypes)
print(df.head())


# Split dataset into train set and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

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
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("LR: ",metrics.accuracy_score(y_test,y_pred))
print("\n")


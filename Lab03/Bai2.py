# Python For Data Science: Lab03
# Ho ten: Ton Thien Minh Anh
# MSSV: 18110049
# Bai 2

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
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB

# Import metrics to evaluate the performance of each model
import sklearn.metrics as metrics

# LOAD DATA
df = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week3/AB_NYC_2019.csv')

# EDA
# print(df.info())

# Find out how many nulls are found in each column in dataset
# print(df.isnull().sum())

# Nhận xét: Sau khi xem qua thống kê dữ liệu, ta thấy rằng các cột "id", "host_name" và "last_review" 
# là các thông tin không cần thiết và không liên quan nhiều đến việc xem xét Data.

# Drop columns
df.drop(["id","host_name","last_review"],axis=1,inplace=True)

# Drop duplicates
rows = df.shape[0]
df.drop_duplicates(subset=df.columns.values[:-1],keep='first',inplace=True)
print(rows-df.shape[0],"duplicated Rows has been removed")

# Drop the rows where at least one elements is missing
df = df.dropna()

# Correlation Map
plt.figure(figsize=(9,9))
sns.heatmap(df.corr(),linewidth=.1,cmap="YlGnBu",annot=True)

# Room Type vs Price
plt.figure(figsize=(15,12))
sns.scatterplot(x='room_type', y='price', data=df)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Room Type vs Price",size=15, weight='bold')

# Nhận xét: Dựa vào biểu đồ quan hệ giữa Room Type và Price
# ta thấy phòng loại "Shared room" luôn có Price < 2000


# Remove Outliers

# Boxplot
plt.subplot(2,2,1)
ax = sns.boxplot(x=df["minimum_nights"])
plt.subplot(2,2,2)
ax = sns.boxplot(x=df["reviews_per_month"])
plt.subplot(2,2,3)
ax = sns.boxplot(x=df["number_of_reviews"])
plt.subplot(2,2,4)
ax = sns.boxplot(x=df["latitude"])

print(df.shape)
Q1 =  df["minimum_nights"].quantile(0.25)
Q3 = df["minimum_nights"].quantile(0.75)
IQR = Q3 - Q1
filter = (df["minimum_nights"] >= Q1 - 1.5 * IQR) & (df["minimum_nights"] <= Q3 + 1.5 *IQR)
df = df.loc[filter]

Q1 =  df["reviews_per_month"].quantile(0.25)
Q3 = df["reviews_per_month"].quantile(0.75)
IQR = Q3 - Q1
filter = (df["reviews_per_month"] >= Q1 - 1.5 * IQR) & (df["reviews_per_month"] <= Q3 + 1.5 *IQR)
df = df.loc[filter]

Q1 =  df["number_of_reviews"].quantile(0.25)
Q3 = df["number_of_reviews"].quantile(0.75)
IQR = Q3 - Q1
filter = (df["number_of_reviews"] >= Q1 - 1.5 * IQR) & (df["number_of_reviews"] <= Q3 + 1.5 *IQR)
df = df.loc[filter]

Q1 =  df["latitude"].quantile(0.25)
Q3 = df["latitude"].quantile(0.75)
IQR = Q3 - Q1
filter = (df["latitude"] >= Q1 - 1.5 * IQR) & (df["latitude"] <= Q3 + 1.5 *IQR)
df = df.loc[filter]

# Convert data to integer or float using Label Encoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for column in df.columns:
    df[column]=labelencoder.fit_transform(df[column])

# SPLIT DATA
X = df.drop(['price'],axis=1)
y = df['price']

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

#Naive Bayes
nb =  GaussianNB()
nb.fit(X_train, y_train)
y_pred=nb.predict(X_test)

print("Accuracy of naive bayees algorithm: ",nb.score(X_test,y_test))

# Normalize
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

#Naive Bayes
nb =  GaussianNB()
nb.fit(normalized_X_train, y_train)
y_pred=nb.predict(X_test)

print("Accuracy of naive bayees algorithm 2: ",nb.score(X_test,y_test))

# Nhận xét: Accuracy sau khi Normalize Data thấp hơn Accuracy trước khi Normalize. Như vậy, đối với Data này, việc Normalize là không hiệu quả và không nên thực hiện


'''
Ho ten: Ton Thien Minh Anh
MSSV: 18110049
Bai thuc hanh Lab04 - Bai 4
'''
# Perform an Exploratory Data Analysis (EDA), Data cleaning, Building clustering models (at least
# three) for prediction, Presenting resultsusing on the datasets in Lab1

# Import thu vien 
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Import Models
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
# Import metrics to evaluate the performance of each model
import sklearn.metrics as metrics


# Load Data
path = '/home/ton-anh/Python Cho KHDL/Week4/xAPI-Edu-Data.csv'
df = pd.read_csv(path)



# =========== EDA ===============

# Xem info data
print(df.info())
# Nhan xet: Sau khi xem qua info, Data khong co feature nao co gia tri Null

# Kiem tra so gia tri NaN
print(df.isna().sum())
# Nhan xet: Data khong co feature nao co gia tri NaN

# Kiem tra so luong tung gia tri trong moi features
for i in range(17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)

# Ve bieu do 

# Ve bieu do countplot cua Class
sns.countplot(x="Class",data=df,linewidth=2,edgecolor=sns.color_palette("dark"))

# Ve Correlation Map
plt.figure(figsize=(8,8))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0)

# Pairplot
sns.pairplot(df,hue='Class')

# =============== DATA CLEANING ==============
# Do data cua chung ta khong co gia tri NaN nen ta chi thuc hien viec xoa bo cac gia tri bi trung nhau:

# Drop duplicates
rows = df.shape[0]
df.drop_duplicates(subset=df.columns.values[:-1],keep='first',inplace=True)
print(rows-df.shape[0],"duplicated Rows has been removed")

# Encoder data
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


# ================== BUILDING MODEL ============

# Split train test 
X = df.drop(['Class'],axis=1)
y = df['Class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

# K - Means Clustering
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

print("K-Means Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# Accuracy ~ 0.1354

# Mean Shift
ms = MeanShift()
ms = ms.fit(X_train)
y_pred = ms.predict(X_test)

print("Mean Shift Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# Accuracy ~ 0.53125

# Gaussian Mixture

gm = GaussianMixture()
gm = gm.fit(X_train)
y_pred = gm.predict(X_test)

print("Gaussian Mixture Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# Accuracy ~ 0.25


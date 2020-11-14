'''
Ho ten: Ton Thien Minh Anh
MSSV: 18110049
Bai thuc hanh Lab04 - Bai 1
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
path = '/home/ton-anh/Python Cho KHDL/Week4/spam.csv'
df = np.genfromtxt(path,delimiter=',')

# =========== EDA ===============

# =============== DATA CLEANING =============

# ================== BUILDING MODEL ============

# Split train test 
X = df[:, :-1]
y = df[:, -1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=17)

# K - Means Clustering
kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

print("K-Means Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# Accuracy ~ 0.6264929

# Mean Shift
ms = MeanShift()
ms = ms.fit(X_train)
y_pred = ms.predict(X_test)

print("Mean Shift Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# Accuracy ~ 0.59066232

# Gaussian Mixture

gm = GaussianMixture()
gm = gm.fit(X_train)
y_pred = gm.predict(X_test)

print("Gaussian Mixture Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("\n")

# Accuracy ~ 0.59391
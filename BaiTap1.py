import numpy as np
import scipy as sp 
import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns 

#Pandas options 
pd.set_option('display.max_colwidth',1000,'display.max_rows',None,\
    'display.max_columns',None)

#Plotting options
# %matplotlib inline
mpl.style.use('ggplot')
sns.set(style='whitegrid')

dataset_pd = pd.read_csv('/home/ton-anh/Python Cho KHDL/Week1/spam.csv')
dataset_np = np.genfromtxt('/home/ton-anh/Python Cho KHDL/Week1/spam.csv',delimiter=',')

print(dataset_pd.shape)
print(dataset_np.shape)

dataset_pd.head()

# Each [ ] corresponding to one row above 
# We're looking at the 5 first rows
dataset_np[0:5,:]

#Separate between feature (X) and label (Y)
X = dataset_np[:,:-1]
Y = dataset_np[:,-1]

print(X.shape)
print(Y.shape)

print(X[0:5,:])
print(Y[0:5])
print(Y[-5:])

from sklearn.model_selection import train_test_split 

# Split dataset
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)
print(X_train.shape,"\n\n",X_test.shape,"\n\n",Y_train.shape,"\n\n",Y_test.shape)

# Import ML (Machine Learning) models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Import libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import metrics to evaluate the performance of each model
import sklearn.metrics as metrics

# clf = Decision Tree Classifier (criterion = 'entrophy')
clf = DecisionTreeClassifier()

# Fit Decision Tree Classifier
clf = clf.fit(X_train,Y_train)
# Predict test set
Y_pred = clf.predict(X_test)
# Evaluate performance of the model
print("CART (Tree Prediction) Accuracy: {}".format(sum(Y_pred == Y_test)/len(Y_pred))) #equivalent to the
print("CART (Tree Prediction) Accuracy by calling metrics: ",metrics.accuracy_score(Y_test,Y_pred))

# Evaluate a score by cross-validation
scores = cross_val_score(clf,X,Y,cv=5)
print("scores={} \n final score = {} \n".format(scores,scores.mean()))
print("\n")

clf = SVC()
# Fit SVM Classifier
clf.fit(X_train,Y_train)
# Predict testset
Y_pred = clf.predict(X_test)
# Evaluate performance of the model
print("SVM Accuracy: ",metrics.accuracy_score(Y_test,Y_pred))
print("\n")
# Evaluate a score by cross-validation
scores = cross_val_score(clf,X,Y,cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))
print("\n")

# Fit Random Forest Classifier
rdf = RandomForestClassifier()
rdf.fit(X_train,Y_train)
# Predict testset
Y_pred = rdf.predict(X_test)
# Evaluate performance of the model 
print("RDF: ",metrics.accuracy_score(Y_test,Y_pred))
print("\n")
#Evaluate a score by cross-validation 
scores=cross_val_score(rdf,X,Y,cv=5)
print("scores={} \n final score = {} \n".format(scores,scores.mean()))
print("\n")


#Fit Logistic Regression Classifier
lr = LogisticRegression()
lr.fit(X_train,Y_train)
#Predict testset
Y_pred = lr.predict(X_test)
# Evaluate performance of the model 
print("LR: ",metrics.accuracy_score(Y_test,Y_pred))
# Evaluate a score by cross-validation
scores = cross_val_score(lr,X,Y,cv=5)
print("scores = {} \n final score = {} \n".format(scores,scores.mean()))
print("\n")
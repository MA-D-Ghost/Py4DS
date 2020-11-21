'''
Ho ten: Ton Thien Minh Anh
MSSV: 18110049
Bai thuc hanh Lab05
'''

# Perform an Exploratory Data Analysis (EDA), Data cleaning, Building models for prediction,
# Presenting resultsusing on the following datasets

# Import thu vien
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# Import Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import metrics to evaluate the performance of each model
import sklearn.metrics as metrics

# Load Data
path = '/home/ton-anh/Python Cho KHDL/Week5/titanic_train.csv'
df = pd.read_csv(path)

# =========== EDA & DATA CLEANING ===============

# In ra man hinh 5 dong dau tien
print(df.head())

# Xem info data
print(df.info())

# Kiem tra so gia tri Null
print(df.isnull().sum())

# Kiem tra ti le gia tri Null trong cot Cabin
percent_missing_cabin = df["Cabin"].isnull().sum()/df.shape[0]*100
print(percent_missing_cabin)
# Nhan xet: Ta thay ti le gia tri Null trong cot Cabin chiem ti le qua lon ~77% => Ta se bo di feature nay khi train model

# Xu ly gia tri Null cua Embarked
print(df['Embarked'].value_counts())
# Nhan xet: Gia tri S chiem ti le nhieu nhat nen ta se thay the cac gia tri Null = S
df['Embarked'] = df['Embarked'].fillna(value='S')

# Xu ly gia tri Null cua Age

print(df['Age'].mean())
df['Age'].plot.hist(rwidth=0.9,color='green')

# Nhan xet: Do cot Age co nhieu gia tri Null, neu ta thay nhung o Null = gia tri mean cua Age se 
# gay nen mat can bang du lieu => Chon gia tri random trong khoang (mean - std, mean + std)

mean = df["Age"].mean()
std = df["Age"].std()
is_null = df["Age"].isnull().sum()
random_values = np.random.randint(mean - std, mean + std, size =is_null)
age_temp = df["Age"].copy()
age_temp[np.isnan(age_temp)] = random_values
df["Age"] = age_temp
df["Age"] = df["Age"].astype(int) 

# Kiem tra lai lan nua sau khi xu ly cac gia tri
print(df.isnull().sum())

# Nhan xet: Ta se loai bo bot nhung cot khong lien quan den data gom "PassengerId", "Name", "Ticket","Cabin"
train_df=df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


# Label Encoder
LE = LabelEncoder()
train_df['Sex'] = LE.fit_transform(train_df['Sex'])
train_df['Embarked'] = LE.fit_transform(train_df['Embarked'])

# Split data
X=train_df.drop(columns=['Survived'])
y=train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5 )

# ============== BUILD MODEL ==============

# Decision Tree
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("DCT Accuracy: ",metrics.accuracy_score(y_test,y_pred))
# Accuracy ~ 0.79888

# Fit Random Forest Classifier

rdf = RandomForestClassifier()
rdf.fit(X_train,y_train)
y_pred = rdf.predict(X_test)

print("RDF: ",metrics.accuracy_score(y_test,y_pred))
# Accuracy ~ 0.837988

#Fit Logistic Regression Classifier
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print("LR: ",metrics.accuracy_score(y_test,y_pred))
# Accuracy ~ 0.810055


# =========== XU LY FILE TITANIC_TEST.CSV ===========
# Load Data
path = '/home/ton-anh/Python Cho KHDL/Week5/titanic_test.csv'
df2 = pd.read_csv(path)

# Xem info data
print(df2.info())

# Kiem tra so gia tri Null
print(df2.isnull().sum())

# Nhan xet: Tuong tu file titanic_train.csv, ta se xu ly cac gia tri Null o Age va Fare

# Xu ly gia tri Null o Age
mean = df2["Age"].mean()
std = df2["Age"].std()
is_null = df2["Age"].isnull().sum()
random_values = np.random.randint(mean - std, mean + std, size =is_null)
age_temp = df2["Age"].copy()
age_temp[np.isnan(age_temp)] = random_values
df2["Age"] = age_temp
df2["Age"] = df2["Age"].astype(int) 

# Xu ly gia tri Null o Fare
mean = df2['Fare'].mean()
df2['Fare'] = df['Fare'].fillna(mean)

# Kiem tra so gia tri Null
print(df2.isnull().sum())

# Nhan xet: Ta se loai bo bot nhung cot khong lien quan den data gom "PassengerId", "Name", "Ticket","Cabin"
df2=df2[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

# Label Encoder
LE = LabelEncoder()
df2['Sex'] = LE.fit_transform(df2['Sex'])
df2['Embarked'] = LE.fit_transform(df2['Embarked'])


# Sau khi xu ly xong, ta co the su dung df2 de test


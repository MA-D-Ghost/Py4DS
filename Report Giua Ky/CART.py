'''
Ho ten: Ton Thien Minh Anh
MSSV: 18110049
Report giua ky: Decision Tree - Classification & Regression (CART)
Dataset used: https://www.kaggle.com/uciml/pima-indians-diabetes-database
'''

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# ===== LOAD DATA =====
path = '/home/ton-anh/Python Cho KHDL/Report Giua Ky/diabetes.csv'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
df = pd.read_csv(path, header = 0, names = col_names)

print(df.head())

# ===== FEATURE SELECTION =====
#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = df[feature_cols] # Features
y = df.label # Target variable

# ===== SPLITTING DATA =====

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test

# ===== BUILDING DECISION TREE MODEL =====

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# ===== EVALUATING MODEL =====

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Accuracy ~ 0.7142857